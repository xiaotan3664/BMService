#include <gtest/gtest.h>
#include "BMPipelinePool.h"

using namespace bm;

using InType = int;
using OutType = int;

class BMPipelineTest : public ::testing::Test {
protected:
    struct Context {
        int index;
    };
    using ContextPtr = std::shared_ptr<Context>;
    using PipelinePool = BMPipelinePool<InType, OutType, Context>;
    std::shared_ptr<PipelinePool> pool;

    void SetUp() override {
        std::function<ContextPtr (size_t)>  contextInitializer = [](size_t i) {
            auto ptr = std::make_shared<Context>();
            ptr->index = i;
            return ptr;
        };
        pool = std::make_shared<PipelinePool>(1, contextInitializer);
        using NodeOut = int;
        std::function<bool (const InType &, NodeOut &, ContextPtr)> func = 
            [](const InType &in, NodeOut &out, ContextPtr) -> bool {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                out = in + 1;
                return true;
            };
        pool->addNode(func);
        pool->addNode(func);
        pool->start();
    }
};

TEST_F(BMPipelineTest, work)
{
    ASSERT_TRUE(pool->push(0));
    int value;
    ASSERT_TRUE(pool->waitAndPop(value));
    ASSERT_EQ(value, 2);
}

TEST_F(BMPipelineTest, join)
{
    size_t round = 10;
    std::thread t([this, round]() {
        for (int i = 0; i < round; ++i)
            pool->push(1);
        pool->join();
    });
    int index, value;
    for (index = 0; pool->waitAndPop(value); ++index);
    ASSERT_EQ(index, round);
    t.join();
}

TEST_F(BMPipelineTest, emptyDeconstruct)
{
    std::function<ContextPtr (size_t)>  contextInitializer = [](size_t i) {
        auto ptr = std::make_shared<Context>();
        ptr->index = i;
        return ptr;
    };
    pool = std::make_shared<PipelinePool>(1, contextInitializer);
    using NodeOut = int;
    std::function<bool (const InType &, NodeOut &, ContextPtr)> func = 
        [](const InType &in, NodeOut &out, ContextPtr) -> bool {
            if (in < 0)
            {
                return false;
            }
            out = in + 1;
            return true;
        };
    pool->addNode(func);
    pool->addNode(func);
    pool->start();
    pool.reset();
}


