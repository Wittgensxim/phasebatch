from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import BoundedSemaphore, RLock
from typing import TypeVar

from .ir_equivalence import EqualityResult
from .runner import release_run_result
from .schema import RunResult


T = TypeVar("T")
EquivalenceKey = tuple[str, str, str]


@dataclass(frozen=True)
class ValidationTransitionKey:
    source_hash: str
    pass_name: str
    pipeline_key: str


@dataclass(frozen=True)
class ValidationTransition:
    ir_path: Path
    canonical_hash: str
    source: str
    run_result: RunResult | None = None


@dataclass(frozen=True)
class ValidationRuntimeSnapshot:
    opt_slot_executions: int
    profile_seed_hits: int
    state_transition_cache_hits: int
    state_transition_cache_misses: int
    state_equivalence_cache_hits: int
    state_equivalence_cache_misses: int
    released_handles: int


class ValidationRuntime:
    def __init__(self, state_dir: Path, max_workers: int):
        self.state_dir = Path(state_dir)
        self.max_workers = max(1, max_workers)
        self._opt_slots = BoundedSemaphore(self.max_workers)
        self.cache_dir = self.state_dir / "artifacts" / "validation_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._transitions: dict[ValidationTransitionKey, ValidationTransition] = {}
        self._transition_inflight: dict[
            ValidationTransitionKey,
            Future[ValidationTransition],
        ] = {}
        self._equivalences: dict[EquivalenceKey, EqualityResult] = {}
        self._equivalence_inflight: dict[
            EquivalenceKey,
            Future[EqualityResult],
        ] = {}
        self._opt_slot_executions = 0
        self._profile_seed_hits = 0
        self._state_transition_cache_hits = 0
        self._state_transition_cache_misses = 0
        self._state_equivalence_cache_hits = 0
        self._state_equivalence_cache_misses = 0
        self._released_handles = 0
        self._closed = False

    def run_with_opt_slot(self, operation: Callable[[], T]) -> T:
        with self._opt_slots:
            with self._lock:
                self._opt_slot_executions += 1
            return operation()

    def get_or_compute_transition(
        self,
        key: ValidationTransitionKey,
        compute: Callable[[Path], ValidationTransition],
    ) -> ValidationTransition:
        with self._lock:
            cached = self._transitions.get(key)
            if cached is not None:
                self._state_transition_cache_hits += 1
                if cached.source == "profile":
                    self._profile_seed_hits += 1
                return cached
            flight = self._transition_inflight.get(key)
            owner = flight is None
            if owner:
                flight = Future()
                self._transition_inflight[key] = flight
                self._state_transition_cache_misses += 1
            else:
                self._state_transition_cache_hits += 1

        if not owner:
            return flight.result()

        try:
            transition = compute(self.transition_cache_path(key))
        except BaseException as exc:
            with self._lock:
                self._transition_inflight.pop(key, None)
            flight.set_exception(exc)
            raise
        with self._lock:
            self._transitions[key] = transition
            self._transition_inflight.pop(key, None)
        flight.set_result(transition)
        return transition

    def transition_cache_path(self, key: ValidationTransitionKey) -> Path:
        digest = hashlib.sha256()
        for value in (key.source_hash, key.pass_name, key.pipeline_key):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, byteorder="big"))
            digest.update(encoded)
        return self.cache_dir / f"{digest.hexdigest()}.ll"

    def seed_transition(
        self,
        key: ValidationTransitionKey,
        transition: ValidationTransition,
    ) -> None:
        with self._lock:
            self._transitions[key] = transition

    def write_keep_marker(self) -> Path:
        marker = self.cache_dir / ".keep_ir_artifacts"
        with self._lock:
            marker.write_text("validation cache retained\n", encoding="utf-8")
        return marker

    def get_or_compute_equivalence(
        self,
        key: EquivalenceKey,
        compute: Callable[[], EqualityResult],
    ) -> EqualityResult:
        normalized_key = self._normalize_equivalence_key(key)
        with self._lock:
            cached = self._equivalences.get(normalized_key)
            if cached is not None:
                self._state_equivalence_cache_hits += 1
                return cached
            flight = self._equivalence_inflight.get(normalized_key)
            owner = flight is None
            if owner:
                flight = Future()
                self._equivalence_inflight[normalized_key] = flight
                self._state_equivalence_cache_misses += 1
            else:
                self._state_equivalence_cache_hits += 1

        if not owner:
            return flight.result()

        try:
            equality = compute()
        except BaseException as exc:
            with self._lock:
                self._equivalence_inflight.pop(normalized_key, None)
            flight.set_exception(exc)
            raise
        with self._lock:
            if equality.tier != "failed":
                self._equivalences[normalized_key] = equality
            self._equivalence_inflight.pop(normalized_key, None)
        flight.set_result(equality)
        return equality

    def snapshot(self) -> ValidationRuntimeSnapshot:
        with self._lock:
            return ValidationRuntimeSnapshot(
                opt_slot_executions=self._opt_slot_executions,
                profile_seed_hits=self._profile_seed_hits,
                state_transition_cache_hits=self._state_transition_cache_hits,
                state_transition_cache_misses=self._state_transition_cache_misses,
                state_equivalence_cache_hits=self._state_equivalence_cache_hits,
                state_equivalence_cache_misses=self._state_equivalence_cache_misses,
                released_handles=self._released_handles,
            )

    def close(self, *, timeout: int = 10) -> int:
        with self._lock:
            if self._closed and not self._transitions:
                return 0
            self._closed = True
            transition_items = list(self._transitions.items())
            self._equivalences.clear()

        groups: dict[int, tuple[RunResult | None, list[ValidationTransitionKey]]] = {}
        for key, transition in transition_items:
            result = transition.run_result
            group_key = id(result) if result is not None else id(transition)
            if group_key not in groups:
                groups[group_key] = (result, [])
            groups[group_key][1].append(key)

        released = 0
        completed_keys: list[ValidationTransitionKey] = []
        errors: list[BaseException] = []
        for result, keys in groups.values():
            completed = (
                result is None
                or result.backend != "worker"
                or not result.module_handle
                or not result.module_handle_owned
            )
            if not completed:
                try:
                    did_release = release_run_result(result, timeout=timeout)
                except BaseException as exc:
                    errors.append(exc)
                else:
                    if did_release:
                        released += 1
                    completed = did_release or not result.module_handle_owned
                    if not completed:
                        errors.append(RuntimeError(f"worker handle release returned false: {result.module_handle}"))
            if completed:
                completed_keys.extend(keys)

        with self._lock:
            for key in completed_keys:
                self._transitions.pop(key, None)
            self._released_handles += released
        if errors:
            raise RuntimeError(
                f"failed to release {len(errors)} validation worker handle(s): {errors[0]}"
            ) from errors[0]
        return released

    def __enter__(self) -> "ValidationRuntime":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    @staticmethod
    def _normalize_equivalence_key(key: EquivalenceKey) -> EquivalenceKey:
        left_hash, right_hash, comparator_version = key
        first_hash, second_hash = sorted((left_hash, right_hash))
        return first_hash, second_hash, comparator_version
