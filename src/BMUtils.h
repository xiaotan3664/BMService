#ifndef BMUTILS_H
#define BMUTILS_H

namespace bm
{
class Uncopiable {
    public:
    Uncopiable() = default;
    Uncopiable(Uncopiable&) = delete;
    Uncopiable(const Uncopiable&) = delete;
    Uncopiable& operator = (const Uncopiable) = delete;
};
}

#endif // BMUTILS_H
