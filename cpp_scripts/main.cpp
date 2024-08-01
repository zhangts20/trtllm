#include <vector>
#include <thread>

#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;


void addRequests(tle::Executor& executor, std::vector<tle::Request>& requests) {
    std::vector<tle::IdType> request_ids;
    for (size_t i = 0; i < requests.size(); ++i) {
        if (executor.canEnqueueRequests()) {
            request_ids.push_back(executor.enqueueRequest(std::move(requests[i])));
        }
    } 
}

int main() {
    initTrtLlmPlugins();

    tle::ExecutorConfig executor_config = tle::ExecutorConfig(1);

    std::string engine_path =
        "/data/models/llama2-7b-fp16-tp1-pp1-256-1024-1024";
    tle::Executor executor = tle::Executor(
        engine_path, tle::ModelType::kDECODER_ONLY, executor_config);

    // params of requests
    tle::VecTokens vec_tokens = {1, 1724, 338, 21784, 29257, 29973};
    tle::SizeType32 max_new_tokens = 17;
    tle::OutputConfig output_config = tle::OutputConfig(false, true, true);
    tle::SamplingConfig sampling_config = tle::SamplingConfig();

    // fake requests
    std::vector<tle::Request> requests;
    for (size_t i = 0; i < 7; ++i) {
        requests.push_back(tle::Request(
            vec_tokens, max_new_tokens + i, true, sampling_config, output_config));
    }

    // enqueue requests
    std::vector<tle::IdType> request_ids;
    for (size_t i = 0; i < requests.size(); ++i) {
        if (executor.canEnqueueRequests()) {
            request_ids.push_back(executor.enqueueRequest(std::move(requests[i])));
        } else {
            return 1;
        }
    }

    std::chrono::milliseconds ms(5000);
    tle::SizeType32 numFinished{0};
    while (numFinished < request_ids.size()) {
        // get results
        std::vector<tle::Response> responses = executor.awaitResponses(ms);
        // loop for each response, if response is finished, print
        for (tle::Response response : responses) {
            tle::Result result = response.getResult();
            // print curr tokens
            auto output_tokens = result.outputTokenIds.at(0);
            printf("Output tokens: %s\n", tlc::vec2str(output_tokens).c_str());
            if (result.isFinal) {
                printf("Finish: %lu\n", response.getRequestId());
                numFinished++;
            }
        }
    }
}
