#include <filesystem>
#include <unordered_map>

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

namespace sfs = std::filesystem;
namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;

struct RequestArgs {
    std::vector<tle::VecTokens> input_ids;
    bool streaming;
    tle::SizeType32 max_new_tokens;
    tle::SizeType32 beam_width;
    tle::SizeType32 top_k;
    tle::FloatType top_p;
    tle::FloatType temperature;
    tle::FloatType repetition_penalty;
    tle::FloatType presence_penalty;
    tle::FloatType frequency_penalty;
    bool return_log_probs;
    bool return_context_logits;
    bool return_generation_logits;
}

struct ModelArgs {

}

class TrtLLM {
public:
    tle::Executor executor;

    TrtLLM(std::string engine_dir, tle::SizeType32 max_beam_width);
    ~TrtLLM() {}
    // read config from config.json
    ModelArgs readConfig(std::string config_path);
    // make requests
    std::vector<tle::Request> makeRequests(RequestArgs request_args);
    // add requests
    std::vector<tle::IdType> enqueueRequests(std::vector<tle::Request> requests);
    // infer requests
    std::unordered_map<tle::IdType, tle::BeamTokens> waitForResponses(std::vector<tle::IdType> request_ids);
};
