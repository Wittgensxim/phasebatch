#include "DifferenceEngine.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>

using namespace llvm;

namespace {

using Clock = std::chrono::steady_clock;

class SilentDiffConsumer final : public Consumer {
public:
  void enterContext(const Value *, const Value *) override {}
  void exitContext() override {}
  void log(StringRef) override { Differences = true; }
  void logf(const LogBuilder &) override { Differences = true; }
  void logd(const DiffLogBuilder &) override { Differences = true; }
  bool hadDifferences() const { return Differences; }

private:
  bool Differences = false;
};

struct StoredModule {
  std::unique_ptr<LLVMContext> Context;
  std::unique_ptr<Module> IR;
};

class Worker {
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
    int64_t RequestID = Request->getInteger("request_id").value_or(-1);
    StringRef Op = Request->getString("op").value_or("");
    if (Op == "shutdown") {
      emit(okResponse(RequestID));
      return false;
    }
    if (Op == "ping") {
      json::Object Response = okResponse(RequestID);
      Response["protocol_version"] = 1;
      Response["llvm_version"] = LLVM_VERSION_STRING;
      Response["module_count"] = static_cast<int64_t>(Modules.size());
      emit(std::move(Response));
      return true;
    }
    if (Op == "load") {
      emit(load(RequestID, *Request));
      return true;
    }
    if (Op == "apply") {
      emit(apply(RequestID, *Request));
      return true;
    }
    if (Op == "materialize") {
      emit(materialize(RequestID, *Request));
      return true;
    }
    if (Op == "compare_paths") {
      emit(comparePaths(RequestID, *Request));
      return true;
    }
    if (Op == "retain") {
      emit(retain(RequestID, *Request));
      return true;
    }
    if (Op == "release") {
      emit(release(RequestID, *Request));
      return true;
    }
    if (Op == "clear") {
      Modules.clear();
      References.clear();
      json::Object Response = okResponse(RequestID);
      Response["released"] = true;
      emit(std::move(Response));
      return true;
    }
    emit(errorResponse(RequestID, "unknown_operation",
                       (Twine("unknown worker operation: ") + Op).str()));
    return true;
  }

private:
  StringMap<std::unique_ptr<StoredModule>> Modules;
  StringMap<uint64_t> References;

  json::Object load(int64_t RequestID, const json::Object &Request) {
    std::optional<StringRef> Path = Request.getString("path");
    if (!Path || Path->empty())
      return errorResponse(RequestID, "invalid_request", "load requires path");
    auto Started = Clock::now();
    auto Context = std::make_unique<LLVMContext>();
    SMDiagnostic Diagnostic;
    std::unique_ptr<Module> M = parseIRFile(*Path, Diagnostic, *Context);
    if (!M) {
      std::string Message;
      raw_string_ostream OS(Message);
      Diagnostic.print("phasebatch-worker", OS);
      OS.flush();
      return errorResponse(RequestID, "parse_failed", Message);
    }
    auto Stored = std::make_unique<StoredModule>();
    Stored->Context = std::move(Context);
    Stored->IR = std::move(M);
    std::string Handle = intern(std::move(Stored));
    json::Object Response =
        moduleResponse(RequestID, Handle, *Modules[Handle]->IR);
    Response["parse_ms"] = elapsedMs(Started);
    Response["cache_hit"] = false;
    return Response;
  }

  json::Object apply(int64_t RequestID, const json::Object &Request) {
    std::optional<StringRef> ParentHandle = Request.getString("parent_handle");
    std::optional<StringRef> Pipeline = Request.getString("pipeline");
    if (!ParentHandle || ParentHandle->empty() || !Pipeline)
      return errorResponse(RequestID, "invalid_request",
                           "apply requires parent_handle and pipeline");
    auto Parent = Modules.find(*ParentHandle);
    if (Parent == Modules.end())
      return errorResponse(RequestID, "unknown_handle",
                           (Twine("unknown module handle: ") + *ParentHandle).str());

    auto TotalStarted = Clock::now();
    auto CloneStarted = Clock::now();
    std::unique_ptr<LLVMContext> ChildContext;
    std::unique_ptr<Module> Child;
    std::string CloneError;
    if (!cloneToFreshContext(*Parent->second->IR, ChildContext, Child,
                             CloneError))
      return errorResponse(RequestID, "clone_failed", CloneError);
    double CloneMS = elapsedMs(CloneStarted);
    bool VerifyEach = Request.getBoolean("verify_each").value_or(true);

    std::string VerifyMessage;
    raw_string_ostream VerifyOS(VerifyMessage);
    auto VerifyStarted = Clock::now();
    if (verifyModule(*Child, &VerifyOS)) {
      VerifyOS.flush();
      return errorResponse(RequestID, "verification_failed", VerifyMessage);
    }
    double VerifyMS = elapsedMs(VerifyStarted);

    double PipelineParseMS = 0.0;
    double PassMS = 0.0;
    {
      LoopAnalysisManager LAM;
      FunctionAnalysisManager FAM;
      CGSCCAnalysisManager CGAM;
      ModuleAnalysisManager MAM;
      PassInstrumentationCallbacks PIC;
      StandardInstrumentations SI(Child->getContext(), false, VerifyEach);
      SI.registerCallbacks(PIC, &MAM);
      PipelineTuningOptions PTO;
      PassBuilder PB(nullptr, PTO, std::nullopt, &PIC);

      TargetLibraryInfoImpl TLII(Triple(Child->getTargetTriple()));
      FAM.registerPass([&] { return TargetLibraryAnalysis(TLII); });
      PB.registerModuleAnalyses(MAM);
      PB.registerCGSCCAnalyses(CGAM);
      PB.registerFunctionAnalyses(FAM);
      PB.registerLoopAnalyses(LAM);
      PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

      ModulePassManager MPM;
      auto PipelineParseStarted = Clock::now();
      if (!Pipeline->empty()) {
        if (Error Err = PB.parsePassPipeline(MPM, *Pipeline)) {
          return errorResponse(RequestID, "invalid_pipeline",
                               toString(std::move(Err)));
        }
      }
      PipelineParseMS = elapsedMs(PipelineParseStarted);
      MPM.addPass(VerifierPass());

      auto PassStarted = Clock::now();
      MPM.run(*Child, MAM);
      PassMS = elapsedMs(PassStarted);
    }

    bool Materialized = false;
    std::string MaterializePath;
    double PrintMS = 0.0;
    if (std::optional<StringRef> Output = Request.getString("materialize_path")) {
      if (!Output->empty()) {
        std::string Error;
        auto PrintStarted = Clock::now();
        if (!writeModule(*Child, *Output, Error))
          return errorResponse(RequestID, "materialize_failed", Error);
        Materialized = true;
        MaterializePath = Output->str();
        PrintMS = elapsedMs(PrintStarted);
      }
    }

    auto Stored = std::make_unique<StoredModule>();
    Stored->Context = std::move(ChildContext);
    Stored->IR = std::move(Child);
    std::string Handle = intern(std::move(Stored));
    json::Object Response =
        moduleResponse(RequestID, Handle, *Modules[Handle]->IR);
    Response["clone_ms"] = CloneMS;
    Response["pipeline_parse_ms"] = PipelineParseMS;
    Response["pass_ms"] = PassMS;
    Response["verify_ms"] = VerifyMS;
    Response["total_ms"] = elapsedMs(TotalStarted);
    Response["materialized"] = Materialized;
    if (Materialized) {
      Response["materialize_path"] = MaterializePath;
      Response["print_ms"] = PrintMS;
    }
    return Response;
  }

  json::Object materialize(int64_t RequestID, const json::Object &Request) {
    std::optional<StringRef> Handle = Request.getString("module_handle");
    std::optional<StringRef> Path = Request.getString("path");
    if (!Handle || !Path || Handle->empty() || Path->empty())
      return errorResponse(RequestID, "invalid_request",
                           "materialize requires module_handle and path");
    auto It = Modules.find(*Handle);
    if (It == Modules.end())
      return errorResponse(RequestID, "unknown_handle",
                           (Twine("unknown module handle: ") + *Handle).str());
    auto Started = Clock::now();
    std::string Error;
    if (!writeModule(*It->second->IR, *Path, Error))
      return errorResponse(RequestID, "materialize_failed", Error);
    json::Object Response = okResponse(RequestID);
    Response["module_handle"] = Handle->str();
    Response["path"] = Path->str();
    Response["print_ms"] = elapsedMs(Started);
    return Response;
  }

  json::Object comparePaths(int64_t RequestID, const json::Object &Request) {
    std::optional<StringRef> LeftPath = Request.getString("left_path");
    std::optional<StringRef> RightPath = Request.getString("right_path");
    if (!LeftPath || !RightPath || LeftPath->empty() || RightPath->empty())
      return errorResponse(RequestID, "invalid_request",
                           "compare_paths requires left_path and right_path");

    auto Started = Clock::now();
    LLVMContext CompareContext;
    SMDiagnostic LeftDiagnostic;
    SMDiagnostic RightDiagnostic;
    std::unique_ptr<Module> Left =
        parseIRFile(*LeftPath, LeftDiagnostic, CompareContext);
    if (!Left) {
      std::string Message;
      raw_string_ostream OS(Message);
      LeftDiagnostic.print("phasebatch-worker", OS);
      OS.flush();
      return errorResponse(RequestID, "parse_failed", Message);
    }
    std::unique_ptr<Module> Right =
        parseIRFile(*RightPath, RightDiagnostic, CompareContext);
    if (!Right) {
      std::string Message;
      raw_string_ostream OS(Message);
      RightDiagnostic.print("phasebatch-worker", OS);
      OS.flush();
      return errorResponse(RequestID, "parse_failed", Message);
    }

    SilentDiffConsumer Consumer;
    DifferenceEngine Engine(Consumer);
    Engine.diff(Left.get(), Right.get());
    json::Object Response = okResponse(RequestID);
    Response["structural_equal"] = !Consumer.hadDifferences();
    Response["compare_ms"] = elapsedMs(Started);
    return Response;
  }

  json::Object release(int64_t RequestID, const json::Object &Request) {
    std::optional<StringRef> Handle = Request.getString("module_handle");
    if (!Handle || Handle->empty())
      return errorResponse(RequestID, "invalid_request",
                           "release requires module_handle");
    auto ModuleIt = Modules.find(*Handle);
    auto ReferenceIt = References.find(*Handle);
    if (ModuleIt == Modules.end() || ReferenceIt == References.end())
      return errorResponse(RequestID, "unknown_handle",
                           (Twine("unknown module handle: ") + *Handle).str());
    uint64_t Remaining = 0;
    if (ReferenceIt->second > 1) {
      Remaining = --ReferenceIt->second;
    } else {
      References.erase(ReferenceIt);
      Modules.erase(ModuleIt);
    }
    json::Object Response = okResponse(RequestID);
    Response["released"] = true;
    Response["module_handle"] = Handle->str();
    Response["remaining_references"] = static_cast<int64_t>(Remaining);
    return Response;
  }

  json::Object retain(int64_t RequestID, const json::Object &Request) {
    std::optional<StringRef> Handle = Request.getString("module_handle");
    if (!Handle || Handle->empty())
      return errorResponse(RequestID, "invalid_request",
                           "retain requires module_handle");
    auto ModuleIt = Modules.find(*Handle);
    auto ReferenceIt = References.find(*Handle);
    if (ModuleIt == Modules.end() || ReferenceIt == References.end())
      return errorResponse(RequestID, "unknown_handle",
                           (Twine("unknown module handle: ") + *Handle).str());
    uint64_t Count = ++ReferenceIt->second;
    json::Object Response = okResponse(RequestID);
    Response["retained"] = true;
    Response["module_handle"] = Handle->str();
    Response["references"] = static_cast<int64_t>(Count);
    return Response;
  }

  std::string intern(std::unique_ptr<StoredModule> Stored) {
    std::string Hash = moduleHash(*Stored->IR);
    std::string Handle = "m-" + Hash;
    if (Modules.find(Handle) == Modules.end()) {
      Modules[Handle] = std::move(Stored);
      References[Handle] = 1;
    } else {
      ++References[Handle];
    }
    return Handle;
  }

  static bool cloneToFreshContext(
      const Module &Source, std::unique_ptr<LLVMContext> &Context,
      std::unique_ptr<Module> &Clone, std::string &ErrorMessage) {
    SmallVector<char, 0> Bitcode;
    raw_svector_ostream Stream(Bitcode);
    WriteBitcodeToFile(Source, Stream);
    auto Buffer = MemoryBuffer::getMemBufferCopy(
        StringRef(Bitcode.data(), Bitcode.size()), Source.getModuleIdentifier());
    Context = std::make_unique<LLVMContext>();
    Expected<std::unique_ptr<Module>> Parsed =
        parseBitcodeFile(Buffer->getMemBufferRef(), *Context);
    if (!Parsed) {
      ErrorMessage = toString(Parsed.takeError());
      Context.reset();
      return false;
    }
    Clone = std::move(*Parsed);
    return true;
  }

  static std::string moduleText(const Module &M) {
    std::string Text;
    raw_string_ostream OS(Text);
    M.print(OS, nullptr);
    OS.flush();
    return Text;
  }

  static std::string moduleHash(const Module &M) {
    std::string Text = moduleText(M);
    SHA256 Hasher;
    Hasher.update(Text);
    std::array<uint8_t, 32> Digest = Hasher.final();
    return toHex(ArrayRef<uint8_t>(Digest), true);
  }

  static json::Object features(const Module &M) {
    int64_t Functions = 0;
    int64_t Blocks = 0;
    int64_t Instructions = 0;
    int64_t Branches = 0;
    int64_t Loads = 0;
    int64_t Stores = 0;
    int64_t Calls = 0;
    int64_t DirectCalls = 0;
    int64_t IntrinsicCalls = 0;
    int64_t IndirectCalls = 0;
    int64_t Phis = 0;
    int64_t Selects = 0;
    int64_t Allocas = 0;
    for (const Function &F : M) {
      if (F.isDeclaration())
        continue;
      ++Functions;
      for (const BasicBlock &BB : F) {
        ++Blocks;
        for (const Instruction &I : BB) {
          ++Instructions;
          if (I.getOpcode() == Instruction::UncondBr ||
              I.getOpcode() == Instruction::CondBr)
            ++Branches;
          if (isa<LoadInst>(I))
            ++Loads;
          if (isa<StoreInst>(I))
            ++Stores;
          if (isa<PHINode>(I))
            ++Phis;
          if (isa<SelectInst>(I))
            ++Selects;
          if (isa<AllocaInst>(I))
            ++Allocas;
          if (const auto *CB = dyn_cast<CallBase>(&I)) {
            ++Calls;
            if (const Function *Callee = CB->getCalledFunction()) {
              if (Callee->isIntrinsic())
                ++IntrinsicCalls;
              else
                ++DirectCalls;
            } else {
              ++IndirectCalls;
            }
          }
        }
      }
    }
    return json::Object{{"functions", Functions},
                        {"basic_blocks", Blocks},
                        {"instructions", Instructions},
                        {"branches", Branches},
                        {"loads", Loads},
                        {"stores", Stores},
                        {"calls", Calls},
                        {"direct_calls", DirectCalls},
                        {"intrinsic_calls", IntrinsicCalls},
                        {"indirect_calls", IndirectCalls},
                        {"phis", Phis},
                        {"selects", Selects},
                        {"allocas", Allocas}};
  }

  static bool writeModule(const Module &M, StringRef Path, std::string &Error) {
    SmallString<256> Parent(Path);
    sys::path::remove_filename(Parent);
    if (!Parent.empty()) {
      std::error_code DirectoryError = sys::fs::create_directories(Parent);
      if (DirectoryError) {
        Error = DirectoryError.message();
        return false;
      }
    }
    std::error_code EC;
    ToolOutputFile Output(Path, EC, sys::fs::OF_Text);
    if (EC) {
      Error = EC.message();
      return false;
    }
    M.print(Output.os(), nullptr);
    Output.keep();
    return true;
  }

  static json::Object moduleResponse(int64_t RequestID, StringRef Handle,
                                     const Module &M) {
    json::Object Response = okResponse(RequestID);
    Response["module_handle"] = Handle.str();
    Response["canonical_hash"] = Handle.drop_front(2).str();
    Response["features"] = features(M);
    return Response;
  }

  static json::Object okResponse(int64_t RequestID) {
    return json::Object{{"request_id", RequestID}, {"status", "ok"}};
  }

  static json::Object errorResponse(int64_t RequestID, StringRef Kind,
                                    StringRef Message) {
    return json::Object{{"request_id", RequestID},
                        {"status", "error"},
                        {"error_kind", Kind},
                        {"error_message", Message}};
  }

  static double elapsedMs(Clock::time_point Started) {
    return std::chrono::duration<double, std::milli>(Clock::now() - Started)
        .count();
  }

  static void emit(json::Object Response) {
    outs() << formatv("{0}\n", json::Value(std::move(Response)));
    outs().flush();
  }
};

} // namespace

int main() {
  Worker W;
  std::string Line;
  while (std::getline(std::cin, Line)) {
    if (!Line.empty() && Line.back() == '\r')
      Line.pop_back();
    if (Line.empty())
      continue;
    if (!W.handleLine(Line))
      break;
  }
  return 0;
}
