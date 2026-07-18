// Strict, experiment-only inspection helper for the advisor's direct-merge
// study.  It intentionally has no production Worker or authority dependency.

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Comdat.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalIFunc.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

struct ParsedModule {
  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<Module> IR;
};

class MergeHelper {
public:
  bool handleLine(StringRef Line) {
    Expected<json::Value> Parsed = json::parse(Line);
    if (!Parsed) {
      emit(errorResponse(-1, "invalid_json", toString(Parsed.takeError())));
      return true;
    }
    json::Object *Request = Parsed->getAsObject();
    if (!Request) {
      emit(errorResponse(-1, "invalid_request", "request must be a JSON object"));
      return true;
    }

    const std::optional<int64_t> RequestID = Request->getInteger("request_id");
    if (!RequestID) {
      emit(errorResponse(-1, "invalid_request",
                         "request_id must be a JSON integer"));
      return true;
    }
    const std::optional<StringRef> Op = Request->getString("op");
    if (!Op || Op->empty()) {
      emit(errorResponse(*RequestID, "invalid_request",
                         "op must be a nonempty JSON string"));
      return true;
    }
    if (*Op == "shutdown") {
      emit(okResponse(*RequestID));
      return false;
    }
    if (*Op == "ping") {
      json::Object Response = okResponse(*RequestID);
      Response["protocol_version"] = 1;
      Response["llvm_version"] = LLVM_VERSION_STRING;
      Response["operations"] =
          json::Array{"ping", "inspect_patch", "merge", "compare_effect"};
      emit(std::move(Response));
      return true;
    }
    if (*Op == "inspect_patch") {
      emit(inspectPatch(*RequestID, *Request));
      return true;
    }
    if (*Op == "merge") {
      emit(merge(*RequestID, *Request));
      return true;
    }
    if (*Op == "compare_effect") {
      emit(compareEffect(*RequestID, *Request));
      return true;
    }

    emit(errorResponse(*RequestID, "unknown_operation",
                       (Twine("unknown merge-helper operation: ") + *Op).str()));
    return true;
  }

private:
  static json::Object inspectPatch(int64_t RequestID, const json::Object &Request) {
    const std::optional<StringRef> BasePath = Request.getString("base_path");
    const std::optional<StringRef> OutputPath = Request.getString("output_path");
    if (!BasePath || !OutputPath || BasePath->empty() || OutputPath->empty()) {
      return errorResponse(RequestID, "invalid_request",
                           "inspect_patch requires base_path and output_path");
    }

    ParsedModule Base;
    std::string ParseError;
    std::string FailureKind;
    if (!parseAndVerify(*BasePath, Base, FailureKind, ParseError)) {
      return errorResponse(RequestID, FailureKind,
                           (Twine("base_path: ") + ParseError).str());
    }
    ParsedModule Output;
    if (!parseAndVerify(*OutputPath, Output, FailureKind, ParseError)) {
      return errorResponse(RequestID, FailureKind,
                           (Twine("output_path: ") + ParseError).str());
    }

    if (Base.IR->getTargetTriple() != Output.IR->getTargetTriple()) {
      return errorResponse(RequestID, "target_triple_changed",
                           "target triple is not a function-body patch");
    }
    if (Base.IR->getDataLayoutStr() != Output.IR->getDataLayoutStr()) {
      return errorResponse(RequestID, "data_layout_changed",
                           "DataLayout is not a function-body patch");
    }

    const std::vector<std::string> BaseInventory = symbolInventory(*Base.IR);
    const std::vector<std::string> OutputInventory = symbolInventory(*Output.IR);
    const std::string BaseInventoryHash = hashStrings(BaseInventory);
    const std::string OutputInventoryHash = hashStrings(OutputInventory);
    if (BaseInventory != OutputInventory) {
      return errorResponse(RequestID, "symbol_inventory_changed",
                           "functions, globals, aliases, or ifuncs changed");
    }

    // Function-level identity is an explicit, typed gate and must run before
    // the generic skeleton comparison.  In particular, deleteBody() makes
    // some definition-only properties invisible in a printed declaration.
    for (const Function &BaseFunction : *Base.IR) {
      const Function *OutputFunction = Output.IR->getFunction(BaseFunction.getName());
      if (!OutputFunction) {
        return errorResponse(RequestID, "symbol_inventory_changed",
                             "output function inventory is incomplete");
      }
      if (functionPolicyIdentity(BaseFunction) !=
          functionPolicyIdentity(*OutputFunction)) {
        return errorResponse(RequestID, "function_identity_changed",
                             "function signature, linkage, attributes, COMDAT, "
                             "personality, GC, or section changed");
      }
    }

    const std::string BaseSkeletonHash = skeletonHash(*Base.IR);
    const std::string OutputSkeletonHash = skeletonHash(*Output.IR);
    if (BaseSkeletonHash != OutputSkeletonHash) {
      return errorResponse(
          RequestID, "module_skeleton_changed",
          "only existing function bodies may differ; module/function identity changed");
    }

    struct ChangedFunction {
      std::string Name;
      std::string BaseHash;
      std::string OutputHash;
    };
    std::vector<ChangedFunction> Changed;
    for (const Function &BaseFunction : *Base.IR) {
      const Function *OutputFunction = Output.IR->getFunction(BaseFunction.getName());
      if (!OutputFunction) {
        return errorResponse(RequestID, "symbol_inventory_changed",
                             "output function inventory is incomplete");
      }
      const std::string BaseHash = isolatedFunctionHash(*Base.IR, BaseFunction.getName());
      const std::string OutputHash =
          isolatedFunctionHash(*Output.IR, OutputFunction->getName());
      if (BaseHash == OutputHash)
        continue;
      if (BaseFunction.isDeclaration() || OutputFunction->isDeclaration()) {
        return errorResponse(RequestID, "declaration_changed",
                             "a declaration cannot be used as a patch target");
      }
      Changed.push_back(
          ChangedFunction{BaseFunction.getName().str(), BaseHash, OutputHash});
    }
    std::sort(Changed.begin(), Changed.end(),
              [](const ChangedFunction &Left, const ChangedFunction &Right) {
                return Left.Name < Right.Name;
              });

    json::Array ChangedHashes;
    json::Array PatchFunctions;
    std::string CanonicalPatch = "advisor_2n_patch_record_v1\n";
    for (const ChangedFunction &Entry : Changed) {
      json::Object HashEntry;
      HashEntry["name"] = Entry.Name;
      HashEntry["base_isolated_hash"] = Entry.BaseHash;
      HashEntry["output_isolated_hash"] = Entry.OutputHash;
      ChangedHashes.push_back(std::move(HashEntry));

      json::Object PatchEntry;
      PatchEntry["name"] = Entry.Name;
      PatchEntry["base_isolated_hash"] = Entry.BaseHash;
      PatchEntry["output_isolated_hash"] = Entry.OutputHash;
      PatchFunctions.push_back(std::move(PatchEntry));

      CanonicalPatch.append(Entry.Name);
      CanonicalPatch.push_back('\n');
      CanonicalPatch.append(Entry.BaseHash);
      CanonicalPatch.push_back('\n');
      CanonicalPatch.append(Entry.OutputHash);
      CanonicalPatch.push_back('\n');
    }

    json::Object PatchRecord;
    PatchRecord["schema_version"] = 1;
    PatchRecord["changed_functions"] = std::move(PatchFunctions);

    json::Object Response = okResponse(RequestID);
    Response["base_module_hash"] = moduleHash(*Base.IR);
    Response["output_module_hash"] = moduleHash(*Output.IR);
    Response["base_skeleton_hash"] = BaseSkeletonHash;
    Response["output_skeleton_hash"] = OutputSkeletonHash;
    Response["base_symbol_inventory_hash"] = BaseInventoryHash;
    Response["output_symbol_inventory_hash"] = OutputInventoryHash;
    Response["changed_functions"] = changedNames(Changed);
    Response["changed_function_hashes"] = std::move(ChangedHashes);
    Response["patch_record"] = std::move(PatchRecord);
    Response["patch_hash"] = sha256(CanonicalPatch);
    return Response;
  }

  // This is deliberately a structured, whole-function replacement.  It never
  // parses or applies a textual diff and it never runs an LLVM pass while
  // merging.  Every source module has first passed inspectPatch() relative to
  // the same base, so the replacement set is a frozen family of body patches.
  static json::Object merge(int64_t RequestID, const json::Object &Request) {
    const std::optional<StringRef> BasePath = Request.getString("base_path");
    const json::Array *OutputPaths = Request.getArray("output_paths");
    const std::optional<StringRef> MergedPath = Request.getString("merged_path");
    if (!BasePath || BasePath->empty() || !OutputPaths || !MergedPath ||
        MergedPath->empty()) {
      return errorResponse(RequestID, "invalid_request",
                           "merge requires base_path, output_paths, and merged_path");
    }

    std::vector<std::string> Paths;
    std::vector<std::string> CanonicalInputPaths;
    Paths.reserve(OutputPaths->size());
    CanonicalInputPaths.reserve(OutputPaths->size() + 1);
    const std::optional<std::string> CanonicalBase = canonicalExistingPath(*BasePath);
    if (!CanonicalBase) {
      return errorResponse(RequestID, "invalid_request",
                           "base_path cannot be canonicalized");
    }
    CanonicalInputPaths.push_back(*CanonicalBase);
    for (const json::Value &Value : *OutputPaths) {
      const std::optional<StringRef> Path = Value.getAsString();
      if (!Path || Path->empty()) {
        return errorResponse(RequestID, "invalid_request",
                             "output_paths must contain only nonempty strings");
      }
      Paths.push_back(Path->str());
      const std::optional<std::string> CanonicalPath = canonicalExistingPath(*Path);
      if (!CanonicalPath) {
        return errorResponse(RequestID, "invalid_request",
                             "output_paths entries cannot be canonicalized");
      }
      CanonicalInputPaths.push_back(*CanonicalPath);
    }
    const std::optional<std::string> CanonicalMerged =
        canonicalOutputPath(*MergedPath);
    if (!CanonicalMerged) {
      return errorResponse(RequestID, "invalid_request",
                           "merged_path cannot be canonicalized");
    }
    if (std::find(CanonicalInputPaths.begin(), CanonicalInputPaths.end(),
                  *CanonicalMerged) != CanonicalInputPaths.end()) {
      return errorResponse(RequestID, "invalid_request",
                           "merged_path must not overwrite a merge input");
    }
    std::sort(Paths.begin(), Paths.end());

    ParsedModule Base;
    std::string FailureKind;
    std::string ErrorMessage;
    if (!parseAndVerify(*BasePath, Base, FailureKind, ErrorMessage)) {
      return errorResponse(RequestID, FailureKind,
                           (Twine("base_path: ") + ErrorMessage).str());
    }
    const std::string BaseModuleHash = moduleHash(*Base.IR);

    struct SourcePatch {
      std::string Path;
      std::vector<std::string> ChangedFunctions;
      std::unique_ptr<Module> Source;
      std::string PatchHash;
      std::string OutputModuleHash;
    };
    std::vector<SourcePatch> Patches;
    std::set<std::string> ClaimedFunctions;
    for (const std::string &Path : Paths) {
      json::Object InspectRequest;
      InspectRequest["base_path"] = BasePath->str();
      InspectRequest["output_path"] = Path;
      json::Object Inspection = inspectPatch(RequestID, InspectRequest);
      if (!responseIsOk(Inspection)) {
        const std::string UnderlyingKind =
            objectString(Inspection, "error_kind").value_or("unknown");
        const std::string UnderlyingMessage =
            objectString(Inspection, "error_message").value_or("patch inspection failed");
        return errorResponse(
            RequestID, "patch_not_mergeable",
            (Twine("output patch ") + Path + " rejected as " + UnderlyingKind +
             ": " + UnderlyingMessage)
                .str());
      }
      const std::optional<std::vector<std::string>> Changed =
          objectStringArray(Inspection, "changed_functions");
      const std::optional<std::string> InspectedBaseHash =
          objectString(Inspection, "base_module_hash");
      const std::optional<std::string> InspectedOutputHash =
          objectString(Inspection, "output_module_hash");
      const std::optional<std::string> InspectedPatchHash =
          objectString(Inspection, "patch_hash");
      if (!Changed || !InspectedBaseHash || !InspectedOutputHash ||
          !InspectedPatchHash) {
        return errorResponse(RequestID, "internal_protocol_error",
                             "inspect_patch success response is incomplete");
      }
      if (*InspectedBaseHash != BaseModuleHash) {
        return errorResponse(RequestID, "patch_changed_during_inspection",
                             "base changed between merge snapshot and inspection");
      }
      // The in-memory source is the only module subsequently cloned.  Its
      // canonical hard hash must equal the exact output snapshot inspected
      // above, otherwise a path replacement/race is fail-closed.
      std::unique_ptr<Module> Source;
      if (!parseAndVerifyInContext(Path, *Base.Context, Source, FailureKind,
                                   ErrorMessage)) {
        return errorResponse(RequestID, "patch_changed_during_inspection",
                             (Twine("cannot read inspected output ") + Path + ": " +
                              FailureKind + ": " + ErrorMessage)
                                 .str());
      }
      if (moduleHash(*Source) != *InspectedOutputHash) {
        return errorResponse(RequestID, "patch_changed_during_inspection",
                             "output changed between inspection and clone snapshot");
      }
      for (const std::string &Name : *Changed) {
        if (!ClaimedFunctions.insert(Name).second) {
          return errorResponse(RequestID, "overlapping_function_patch",
                               (Twine("multiple patches modify function ") + Name).str());
        }
      }
      Patches.push_back(SourcePatch{Path, *Changed, std::move(Source),
                                    *InspectedPatchHash, *InspectedOutputHash});
    }

    const auto Start = std::chrono::steady_clock::now();
    std::unique_ptr<Module> Merged = CloneModule(*Base.IR);
    for (const SourcePatch &Patch : Patches) {
      // CloneFunctionInto operates on LLVM values, whose type identity is
      // context-local.  Source is the hash-validated in-memory snapshot read
      // above, not a second read of Patch.Path.
      for (const std::string &Name : Patch.ChangedFunctions) {
        Function *TargetFunction = Merged->getFunction(Name);
        const Function *SourceFunction = Patch.Source->getFunction(Name);
        if (!TargetFunction || !SourceFunction || TargetFunction->isDeclaration() ||
            SourceFunction->isDeclaration()) {
          return errorResponse(RequestID, "merge_invalid",
                               (Twine("missing definition for changed function ") + Name).str());
        }
        if (Error CloneError = replaceFunctionBody(*TargetFunction, *SourceFunction,
                                                   *Merged, *Patch.Source)) {
          return errorResponse(RequestID, "merge_invalid",
                               (Twine("cannot replace function ") + Name + ": " +
                                toString(std::move(CloneError)))
                                   .str());
        }
      }
    }

    std::string VerifyError;
    raw_string_ostream VerifyOS(VerifyError);
    if (verifyModule(*Merged, &VerifyOS)) {
      VerifyOS.flush();
      return errorResponse(RequestID, "merge_invalid",
                           (Twine("merged module failed verification: ") + VerifyError).str());
    }
    VerifyOS.flush();
    const bool SameInventory = symbolInventory(*Merged) == symbolInventory(*Base.IR);
    const std::string BaseSkeleton = skeletonHash(*Base.IR);
    const std::string MergedSkeleton = skeletonHash(*Merged);
    if (!SameInventory || MergedSkeleton != BaseSkeleton) {
      const std::string BaseSkeletonText = skeletonText(*Base.IR);
      const std::string MergedSkeletonText = skeletonText(*Merged);
      return errorResponse(RequestID, "merge_invalid",
                           (Twine("whole-function replacement changed module structure; ") +
                            "symbol_inventory_equal=" + (SameInventory ? "true" : "false") +
                            "; base_skeleton=" + BaseSkeleton +
                            "; merged_skeleton=" + MergedSkeleton + "; " +
                            skeletonDifference(BaseSkeletonText, MergedSkeletonText))
                               .str());
    }

    const std::string MergedText = moduleText(*Merged);
    std::error_code EC;
    raw_fd_ostream Output(*MergedPath, EC, sys::fs::OF_None);
    if (EC) {
      return errorResponse(RequestID, "output_write_failed",
                           (Twine("cannot write merged_path: ") + EC.message()).str());
    }
    Output << MergedText;
    Output.flush();
    if (Output.has_error()) {
      return errorResponse(RequestID, "output_write_failed",
                           "failed while writing merged_path");
    }

    ParsedModule Written;
    if (!parseAndVerify(*MergedPath, Written, FailureKind, ErrorMessage)) {
      return errorResponse(RequestID, "merge_invalid",
                           (Twine("written merged module is invalid: ") + FailureKind +
                            ": " + ErrorMessage)
                               .str());
    }
    const std::string MergedHash = moduleHash(*Merged);
    if (MergedHash != moduleHash(*Written.IR)) {
      return errorResponse(RequestID, "merge_invalid",
                           "written merged module hash differs after parse/verify round-trip");
    }

    std::vector<std::string> Contributed(ClaimedFunctions.begin(), ClaimedFunctions.end());
    std::vector<std::string> InputPatchHashes;
    std::vector<std::string> InputOutputModuleHashes;
    InputPatchHashes.reserve(Patches.size());
    InputOutputModuleHashes.reserve(Patches.size());
    for (const SourcePatch &Patch : Patches) {
      InputPatchHashes.push_back(Patch.PatchHash);
      InputOutputModuleHashes.push_back(Patch.OutputModuleHash);
    }
    std::sort(InputPatchHashes.begin(), InputPatchHashes.end());
    std::sort(InputOutputModuleHashes.begin(), InputOutputModuleHashes.end());
    json::Object Response = okResponse(RequestID);
    Response["base_module_hash"] = moduleHash(*Base.IR);
    Response["base_skeleton_hash"] = skeletonHash(*Base.IR);
    Response["output_module_hash"] = MergedHash;
    Response["output_skeleton_hash"] = skeletonHash(*Written.IR);
    Response["merged_functions"] = stringArray(Contributed);
    Response["contributed_functions"] = stringArray(Contributed);
    Response["input_patch_hashes"] = stringArray(InputPatchHashes);
    Response["input_output_module_hashes"] = stringArray(InputOutputModuleHashes);
    Response["merge_input_count"] = static_cast<int64_t>(Paths.size());
    Response["merge_wall_time_ns"] = static_cast<int64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - Start)
            .count());
    return Response;
  }

  // Equality is intentionally stronger than same changed names or final module
  // hash.  The canonical patch hash fixes the changed-function hashes, while
  // the explicit protected-function comparison checks that the other direct
  // merge contributions survived Pi's second-round execution unchanged.
  static json::Object compareEffect(int64_t RequestID, const json::Object &Request) {
    const std::optional<StringRef> FirstBase = Request.getString("first_base_path");
    const std::optional<StringRef> FirstOutput = Request.getString("first_output_path");
    const std::optional<StringRef> SecondBase = Request.getString("second_base_path");
    const std::optional<StringRef> SecondOutput = Request.getString("second_output_path");
    const json::Array *ProtectedValues = Request.getArray("protected_functions");
    if (!FirstBase || FirstBase->empty() || !FirstOutput || FirstOutput->empty() ||
        !SecondBase || SecondBase->empty() || !SecondOutput || SecondOutput->empty() ||
        !ProtectedValues) {
      return errorResponse(RequestID, "invalid_request",
                           "compare_effect requires four paths and protected_functions");
    }

    std::vector<std::string> Protected;
    Protected.reserve(ProtectedValues->size());
    for (const json::Value &Value : *ProtectedValues) {
      const std::optional<StringRef> Name = Value.getAsString();
      if (!Name || Name->empty()) {
        return errorResponse(RequestID, "invalid_request",
                             "protected_functions must contain only nonempty strings");
      }
      Protected.push_back(Name->str());
    }
    std::sort(Protected.begin(), Protected.end());
    if (std::adjacent_find(Protected.begin(), Protected.end()) != Protected.end()) {
      return errorResponse(RequestID, "invalid_request",
                           "protected_functions must not contain duplicates");
    }

    json::Object FirstRequest;
    FirstRequest["base_path"] = FirstBase->str();
    FirstRequest["output_path"] = FirstOutput->str();
    json::Object First = inspectPatch(RequestID, FirstRequest);
    if (!responseIsOk(First))
      return wrappedInspectionError(RequestID, "first_patch_invalid", First);

    // The helper derives the complete contribution family from the actual
    // merged second-round input.  A caller cannot weaken the preservation
    // check by omitting one of the other N-1 first-round contributions.
    json::Object ContributionRequest;
    ContributionRequest["base_path"] = FirstBase->str();
    ContributionRequest["output_path"] = SecondBase->str();
    json::Object Contributions = inspectPatch(RequestID, ContributionRequest);
    if (!responseIsOk(Contributions))
      return wrappedInspectionError(RequestID, "merged_input_invalid", Contributions);
    const std::optional<std::vector<std::string>> ExpectedProtected =
        objectStringArray(Contributions, "changed_functions");
    if (!ExpectedProtected) {
      return errorResponse(RequestID, "internal_protocol_error",
                           "merged input inspection lacks changed_functions");
    }
    if (Protected != *ExpectedProtected) {
      return errorResponse(
          RequestID, "protected_functions_mismatch",
          "protected_functions must exactly match first_base to second_base contributions");
    }

    json::Object SecondRequest;
    SecondRequest["base_path"] = SecondBase->str();
    SecondRequest["output_path"] = SecondOutput->str();
    json::Object Second = inspectPatch(RequestID, SecondRequest);
    if (!responseIsOk(Second))
      return wrappedInspectionError(RequestID, "second_patch_invalid", Second);

    const std::optional<std::vector<std::string>> FirstChanged =
        objectStringArray(First, "changed_functions");
    const std::optional<std::vector<std::string>> SecondChanged =
        objectStringArray(Second, "changed_functions");
    const std::optional<std::string> FirstPatchHash = objectString(First, "patch_hash");
    const std::optional<std::string> SecondPatchHash = objectString(Second, "patch_hash");
    const std::optional<std::string> FirstBaseSkeleton =
        objectString(First, "base_skeleton_hash");
    const std::optional<std::string> FirstOutputSkeleton =
        objectString(First, "output_skeleton_hash");
    const std::optional<std::string> SecondBaseSkeleton =
        objectString(Second, "base_skeleton_hash");
    const std::optional<std::string> SecondOutputSkeleton =
        objectString(Second, "output_skeleton_hash");
    const std::optional<std::string> FirstInventory =
        objectString(First, "base_symbol_inventory_hash");
    const std::optional<std::string> SecondInventory =
        objectString(Second, "base_symbol_inventory_hash");
    if (!FirstChanged || !SecondChanged || !FirstPatchHash || !SecondPatchHash ||
        !FirstBaseSkeleton || !FirstOutputSkeleton || !SecondBaseSkeleton ||
        !SecondOutputSkeleton || !FirstInventory || !SecondInventory) {
      return errorResponse(RequestID, "internal_protocol_error",
                           "inspect_patch success response is incomplete");
    }

    ParsedModule ParsedSecondBase;
    ParsedModule ParsedSecondOutput;
    std::string FailureKind;
    std::string ErrorMessage;
    if (!parseAndVerify(*SecondBase, ParsedSecondBase, FailureKind, ErrorMessage) ||
        !parseAndVerify(*SecondOutput, ParsedSecondOutput, FailureKind, ErrorMessage)) {
      return errorResponse(RequestID, "second_patch_invalid",
                           (Twine("cannot re-parse second-round modules: ") + FailureKind +
                            ": " + ErrorMessage)
                               .str());
    }
    bool ProtectedPreserved = true;
    for (const std::string &Name : Protected) {
      const Function *Before = ParsedSecondBase.IR->getFunction(Name);
      const Function *After = ParsedSecondOutput.IR->getFunction(Name);
      if (!Before || !After || Before->isDeclaration() || After->isDeclaration()) {
        return errorResponse(RequestID, "protected_function_missing",
                             (Twine("protected function is unavailable: ") + Name).str());
      }
      if (isolatedFunctionHash(*ParsedSecondBase.IR, Name) !=
          isolatedFunctionHash(*ParsedSecondOutput.IR, Name)) {
        ProtectedPreserved = false;
      }
    }

    const bool SkeletonsUnchanged =
        *FirstBaseSkeleton == *FirstOutputSkeleton &&
        *FirstBaseSkeleton == *SecondBaseSkeleton &&
        *FirstBaseSkeleton == *SecondOutputSkeleton;
    const bool InventoriesUnchanged =
        *FirstInventory == *SecondInventory &&
        objectString(First, "output_symbol_inventory_hash") == FirstInventory &&
        objectString(Second, "output_symbol_inventory_hash") == SecondInventory;
    const bool SameEffect = *FirstChanged == *SecondChanged &&
                            *FirstPatchHash == *SecondPatchHash &&
                            SkeletonsUnchanged && InventoriesUnchanged &&
                            ProtectedPreserved;

    json::Object Response = okResponse(RequestID);
    Response["same_effect"] = SameEffect;
    Response["first_changed_functions"] = stringArray(*FirstChanged);
    Response["second_changed_functions"] = stringArray(*SecondChanged);
    Response["first_patch_hash"] = *FirstPatchHash;
    Response["second_patch_hash"] = *SecondPatchHash;
    Response["protected_functions"] = stringArray(Protected);
    Response["expected_protected_functions"] = stringArray(*ExpectedProtected);
    Response["protected_functions_preserved"] = ProtectedPreserved;
    Response["skeletons_unchanged"] = SkeletonsUnchanged;
    Response["symbol_inventories_unchanged"] = InventoriesUnchanged;
    return Response;
  }

  static Error replaceFunctionBody(Function &Target, const Function &Source,
                                   Module &TargetModule, const Module &SourceModule) {
    if (typeText(Target.getFunctionType()) != typeText(Source.getFunctionType()) ||
        functionPolicyIdentity(Target) != functionPolicyIdentity(Source)) {
      return createStringError(inconvertibleErrorCode(),
                               "function identity mismatch");
    }
    ValueToValueMapTy VMap;
    for (const GlobalValue &SourceGlobal : SourceModule.global_values()) {
      GlobalValue *TargetGlobal = TargetModule.getNamedValue(SourceGlobal.getName());
      if (!TargetGlobal ||
          typeText(TargetGlobal->getValueType()) != typeText(SourceGlobal.getValueType())) {
        return createStringError(inconvertibleErrorCode(),
                                 "global mapping mismatch");
      }
      VMap[&SourceGlobal] = TargetGlobal;
    }
    auto TargetArgument = Target.arg_begin();
    for (const Argument &SourceArgument : Source.args()) {
      if (TargetArgument == Target.arg_end()) {
        return createStringError(inconvertibleErrorCode(), "argument count mismatch");
      }
      VMap[&SourceArgument] = &*TargetArgument++;
    }
    if (TargetArgument != Target.arg_end()) {
      return createStringError(inconvertibleErrorCode(), "argument count mismatch");
    }
    Target.deleteBody();
    SmallVector<ReturnInst *, 8> Returns;
    CloneFunctionInto(&Target, &Source, VMap,
                      // All global values have an explicit same-context map.
                      // ClonedModule avoids synthesizing !llvm.dbg.cu (a
                      // forbidden module-level change) during a body clone.
                      CloneFunctionChangeType::ClonedModule, Returns);
    return Error::success();
  }

  static bool responseIsOk(const json::Object &Response) {
    const std::optional<StringRef> Status = Response.getString("status");
    return Status && *Status == "ok";
  }

  static std::optional<std::string> objectString(const json::Object &Object,
                                                  StringRef Key) {
    const std::optional<StringRef> Value = Object.getString(Key);
    if (!Value)
      return std::nullopt;
    return Value->str();
  }

  static std::optional<std::vector<std::string>>
  objectStringArray(const json::Object &Object, StringRef Key) {
    const json::Array *Values = Object.getArray(Key);
    if (!Values)
      return std::nullopt;
    std::vector<std::string> Result;
    Result.reserve(Values->size());
    for (const json::Value &Value : *Values) {
      const std::optional<StringRef> StringValue = Value.getAsString();
      if (!StringValue)
        return std::nullopt;
      Result.push_back(StringValue->str());
    }
    return Result;
  }

  static json::Array stringArray(const std::vector<std::string> &Values) {
    json::Array Result;
    for (const std::string &Value : Values)
      Result.push_back(Value);
    return Result;
  }

  static json::Object wrappedInspectionError(int64_t RequestID, StringRef Kind,
                                             const json::Object &Inspection) {
    const std::string UnderlyingKind =
        objectString(Inspection, "error_kind").value_or("unknown");
    const std::string UnderlyingMessage =
        objectString(Inspection, "error_message").value_or("patch inspection failed");
    return errorResponse(RequestID, Kind,
                         (Twine("inspect_patch rejected input as ") + UnderlyingKind +
                          ": " + UnderlyingMessage)
                             .str());
  }

  static bool parseAndVerify(StringRef Path, ParsedModule &Parsed,
                              std::string &FailureKind,
                              std::string &ErrorMessage) {
    FailureKind.clear();
    ErrorMessage.clear();
    Parsed.Context = std::make_unique<LLVMContext>();
    if (!parseAndVerifyInContext(Path, *Parsed.Context, Parsed.IR, FailureKind,
                                 ErrorMessage)) {
      Parsed.Context.reset();
      return false;
    }
    return true;
  }

  static bool parseAndVerifyInContext(StringRef Path, LLVMContext &Context,
                                      std::unique_ptr<Module> &IR,
                                      std::string &FailureKind,
                                      std::string &ErrorMessage) {
    FailureKind.clear();
    ErrorMessage.clear();
    SMDiagnostic Diagnostic;
    IR = parseIRFile(Path, Diagnostic, Context);
    if (!IR) {
      raw_string_ostream OS(ErrorMessage);
      Diagnostic.print("phasebatch-2n-merge", OS);
      OS.flush();
      FailureKind = "parse_failed";
      return false;
    }
    raw_string_ostream VerifyOS(ErrorMessage);
    if (verifyModule(*IR, &VerifyOS)) {
      VerifyOS.flush();
      IR.reset();
      FailureKind = "verification_failed";
      return false;
    }
    VerifyOS.flush();
    return true;
  }

  static std::string moduleText(const Module &M) {
    // parseIRFile uses the path as the module identifier.  That provenance is
    // not IR state and must not turn equivalent base/output modules into a
    // false patch conflict.
    std::unique_ptr<Module> Clone = CloneModule(M);
    Clone->setModuleIdentifier("advisor-2n-canonical-module");
    std::string Text;
    raw_string_ostream OS(Text);
    Clone->print(OS, nullptr);
    OS.flush();
    return Text;
  }

  static std::string sha256(StringRef Text) {
    SHA256 Hasher;
    Hasher.update(Text);
    const auto Digest = Hasher.final();
    return toHex(ArrayRef<uint8_t>(Digest), true);
  }

  static std::string moduleHash(const Module &M) { return sha256(moduleText(M)); }

  static std::optional<std::string> canonicalExistingPath(StringRef Path) {
    SmallString<256> Canonical;
    if (sys::fs::real_path(Path, Canonical))
      return std::nullopt;
    return Canonical.str().str();
  }

  static std::optional<std::string> canonicalOutputPath(StringRef Path) {
    if (const std::optional<std::string> Existing = canonicalExistingPath(Path))
      return Existing;

    SmallString<256> Parent(Path);
    const std::string Filename = sys::path::filename(Parent).str();
    if (Filename.empty())
      return std::nullopt;
    sys::path::remove_filename(Parent);
    if (Parent.empty())
      Parent = ".";
    SmallString<256> CanonicalParent;
    if (sys::fs::real_path(Parent, CanonicalParent))
      return std::nullopt;
    sys::path::append(CanonicalParent, Filename);
    return CanonicalParent.str().str();
  }

  static std::string skeletonHash(const Module &M) {
    return sha256(skeletonText(M));
  }

  static std::string skeletonText(const Module &M) {
    std::unique_ptr<Module> Clone = CloneModule(M);
    for (Function &Function : *Clone) {
      if (!Function.isDeclaration())
        Function.deleteBody();
    }
    return moduleText(*Clone);
  }

  static std::string skeletonDifference(StringRef Left, StringRef Right) {
    const size_t Limit = std::min(Left.size(), Right.size());
    size_t Index = 0;
    while (Index < Limit && Left[Index] == Right[Index])
      ++Index;
    const size_t Start = Index > 40 ? Index - 40 : 0;
    const size_t LeftLength = std::min<size_t>(80, Left.size() - Start);
    const size_t RightLength = std::min<size_t>(80, Right.size() - Start);
    return (Twine("first_difference_at=") + Twine(Index) + "; base_fragment=" +
            Left.substr(Start, LeftLength) + "; merged_fragment=" +
            Right.substr(Start, RightLength))
        .str();
  }

  static std::string isolatedFunctionHash(const Module &M, StringRef Name) {
    std::unique_ptr<Module> Clone = CloneModule(M);
    for (Function &Function : *Clone) {
      if (Function.getName() != Name && !Function.isDeclaration())
        Function.deleteBody();
    }
    return moduleHash(*Clone);
  }

  static std::string typeText(Type *TypeValue) {
    std::string Text;
    raw_string_ostream OS(Text);
    TypeValue->print(OS);
    OS.flush();
    return Text;
  }

  static std::string functionPolicyIdentity(const Function &Function) {
    std::string Text;
    raw_string_ostream OS(Text);
    OS << Function.getName() << '\n';
    OS << typeText(Function.getFunctionType()) << '\n';
    OS << static_cast<unsigned>(Function.getLinkage()) << '\n';
    OS << static_cast<unsigned>(Function.getVisibility()) << '\n';
    OS << static_cast<unsigned>(Function.getDLLStorageClass()) << '\n';
    OS << static_cast<unsigned>(Function.getUnnamedAddr()) << '\n';
    OS << static_cast<unsigned>(Function.getCallingConv()) << '\n';
    Function.getAttributes().print(OS);
    OS << '\n';
    if (Function.hasComdat())
      OS << Function.getComdat()->getName();
    OS << '\n';
    if (Function.hasPersonalityFn())
      Function.getPersonalityFn()->printAsOperand(OS, false);
    OS << '\n' << Function.getGC() << '\n' << Function.getSection() << '\n';
    OS << Function.getPartition() << '\n';
    return Text;
  }

  static std::vector<std::string> symbolInventory(const Module &M) {
    std::vector<std::string> Inventory;
    for (const Function &Function : M) {
      Inventory.push_back((Twine("function\t") + Function.getName() + "\t" +
                           typeText(Function.getFunctionType()) + "\t" +
                           (Function.isDeclaration() ? "declaration" : "definition"))
                              .str());
    }
    for (const GlobalVariable &Global : M.globals()) {
      Inventory.push_back((Twine("global\t") + Global.getName() + "\t" +
                           typeText(Global.getValueType()))
                              .str());
    }
    for (const GlobalAlias &Alias : M.aliases()) {
      Inventory.push_back((Twine("alias\t") + Alias.getName() + "\t" +
                           typeText(Alias.getValueType()))
                              .str());
    }
    for (const GlobalIFunc &IFunc : M.ifuncs()) {
      Inventory.push_back((Twine("ifunc\t") + IFunc.getName() + "\t" +
                           typeText(IFunc.getValueType()))
                              .str());
    }
    std::sort(Inventory.begin(), Inventory.end());
    return Inventory;
  }

  static std::string hashStrings(const std::vector<std::string> &Strings) {
    std::string Joined;
    for (const std::string &Value : Strings) {
      Joined.append(Value);
      Joined.push_back('\n');
    }
    return sha256(Joined);
  }

  template <typename ChangedFunctionT>
  static json::Array changedNames(const std::vector<ChangedFunctionT> &Changed) {
    json::Array Names;
    for (const ChangedFunctionT &Entry : Changed)
      Names.push_back(Entry.Name);
    return Names;
  }

  static json::Object okResponse(int64_t RequestID) {
    return json::Object{{"request_id", RequestID}, {"status", "ok"}};
  }

  static json::Object errorResponse(int64_t RequestID, StringRef Kind,
                                    StringRef Message) {
    return json::Object{{"request_id", RequestID},
                        {"status", "error"},
                        {"error_kind", Kind.str()},
                        {"error_message", Message.str()}};
  }

  static void emit(json::Object Response) {
    outs() << formatv("{0}\n", json::Value(std::move(Response)));
    outs().flush();
  }
};

} // namespace

int main() {
  MergeHelper Helper;
  std::string Line;
  while (std::getline(std::cin, Line)) {
    if (!Line.empty() && Line.back() == '\r')
      Line.pop_back();
    if (Line.empty())
      continue;
    if (!Helper.handleLine(Line))
      break;
  }
  return 0;
}
