#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include "BMQueue.h"

class BMQueueTest : public ::testing::Test {
protected:
    bm::BMQueue<int> q0;
    std::vector<std::thread> threads;
    void sendAfterDelay(int value, int ms) {
        threads.push_back(std::thread(
            [this, ms, value]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                this->q0.push(value);
            }));
    }
    void TearDown() override {
        for (auto &t : threads)
        {
            if (t.joinable())
                t.join();
        }
    }
};

TEST_F(BMQueueTest, pushAndPop)
{
    sendAfterDelay(1, 100);
    int value;
    ASSERT_TRUE(q0.waitAndPop(value));
    ASSERT_EQ(value, 1);
    ASSERT_TRUE(q0.empty());
}
