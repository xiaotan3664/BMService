#ifndef BMQUEUE_H
#define BMQUEUE_H
#include <memory>
#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include "BMUtils.h"
namespace bm {
#define LOCK(name) std::lock_guard<std::mutex> guard(name##_mutex)

class BMQueueVoid {
public:
    virtual ~BMQueueVoid() {};
};

template <typename T>
class BMQueue: public Uncopiable, public BMQueueVoid
{
private:
    struct Node {
        std::shared_ptr<T> data;
        std::unique_ptr<Node> next;
    };
    std::mutex head_mutex;
    std::unique_ptr<Node> head;
    std::mutex tail_mutex;
    Node* tail;
    std::condition_variable data_cond;
    Node* getTail(){
        LOCK(tail);
        return tail;
    }
    std::unique_ptr<Node> popHead(){
        auto old_head = std::move(head);
        head = std::move(old_head->next);
        return old_head;
    }

    std::unique_lock<std::mutex> waitForData(){
        std::unique_lock<std::mutex> ulock(head_mutex);
        data_cond.wait(ulock, [&]{ return head.get() == tail; });
        return ulock;
    }

    std::unique_ptr<Node> waitPopHead(){
        std::unique_lock<std::mutex> head_lock(waitForData());
        return popHead();
    }

    std::unique_ptr<Node> tryPopHead(){
        LOCK(head);
        if (head.get() == getTail()){
            return std::unique_ptr<Node>();
        }
        return popHead();
    }

public:
    BMQueue(): head(new Node), tail(head.get()){}

    std::shared_ptr<T> tryPop() {
        auto oldHead = tryPopHead();
        return oldHead? oldHead->data: std::shared_ptr<T>();
    }

    bool tryPop(T& value){
        auto oldHead = tryPopHead();
        if(oldHead){
            value = std::move(*oldHead->data);
            return true;
        }
        return false;
    }

    std::shared_ptr<T> waitAndPop() {
        const auto oldHead = waitPopHead();
        return oldHead->data;
    }

    void waitAndPop(T& value) {
        const auto oldHead = waitPopHead();
        value = std::move(*oldHead->data);
    }

    void push(T new_value) {
        std::shared_ptr<T> new_data(
                    std::make_shared<T>(std::move(new_value)));
        std::unique_ptr<Node> new_node(new Node);

        {
            LOCK(tail);
            tail->data = new_data;
            auto new_tail = new_node.get();
            tail->next = std::move(new_node);
            tail = new_tail;
        }
        data_cond.notify_one();
    }

    bool empty() {
        LOCK(head);
        return head.get() == getTail();
    }
};

template<typename T>
class BMWorkStealingQueue: public Uncopiable
{
private:
    std::deque<T> the_queue;
    mutable std::mutex queue_mutex;
public:
    BMWorkStealingQueue() {}

    void push(T data){
        LOCK(queue);
        the_queue.push_front(std::move(data));
    }

    bool empty() const {
        LOCK(queue);
        return the_queue.empty();
    }

    bool tryPop(T& res) {
        LOCK(queue);
        if(the_queue.empty()){
            return false;
        }
        res = std::move(the_queue.front());
        the_queue.pop_front();
        return true;
    }

    // steal item from back
    bool trySteal(T& res){
        LOCK(queue);
        if(the_queue.empty()){
            return false;
        }
        res = std::move(the_queue.back());
        the_queue.pop_back();
        return true;
    }

};

#undef LOCK
}
#endif // BMQUEUE_H
