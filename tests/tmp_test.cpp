#include <nlohmann/json.hpp>

#include <iostream>
#include <memory>
#include <utility>
#include <vector>

struct up1 {
    up1() { std::cout << "up1 default ctor" << std::endl; }
    up1(int x) : x { x } { std::cout << "up1 ctor" << std::endl; }
    up1(const up1& other) : x { other.x } { std::cout << "up1 copy ctor" << std::endl; }
    up1(up1&& other) noexcept : x { other.x } { std::cout << "up1 move ctor" << std::endl; }
    int x = 1;
};

struct up2 {
    up2() { std::cout << "up2 default ctor" << std::endl; }
    up2(int x) : x { x } { std::cout << "up2 ctor" << std::endl; }
    up2(const up2& other) : x { other.x } { std::cout << "up2 copy ctor" << std::endl; }
    up2(up2&& other) noexcept : x { other.x } { std::cout << "up2 move ctor" << std::endl; }
    int x = 2;
};

struct serializer {
    template <typename T>
    void serialize(const T& t) {
        std::cout << "serialize: " << t.x << std::endl;
    }
};

struct serializer2 {
    template <typename T>
    void serialize(const T& t) {
        std::cout << "serialize 2: " << t.x << std::endl;
    }
};

template <typename S>
class update {
    struct update_concept {
        virtual ~update_concept() = default;
        virtual void do_serialize(S& s) const = 0;
    };

    template <typename T>
    struct update_model : update_concept {
        update_model(T t) : up { std::move(t) } {}
        void do_serialize(S& s) const override { s.serialize(up); }
        T up;
    };

    friend void serialize(S& s, const update& up) { up.pimpl->do_serialize(s); }

    std::shared_ptr<update_concept> pimpl;

public:
    template <typename T>
    update(T t) : pimpl { std::make_shared<update_model<T>>(std::move(t)) } {}

    template <typename T>
    T& as() {
        if (auto ptr = dynamic_cast<update_model<T>*>(pimpl.get())) {
            return ptr->up;
        } else {
            throw std::runtime_error("bad cast");
        }
    }

    template <typename T>
    [[nodiscard]] const T& as() const {
        return dynamic_cast<update_model<T>*>(pimpl.get())->up;
    }
};

int main() {
    try {
        std::vector<update<serializer2>> updates;
        updates.emplace_back(up1 {});
        updates.emplace_back(up2 { 100 });

        serializer2 s{};

        for (const auto& up : updates) {
            serialize(s, up);
        }

        auto& u1 = updates[0].as<up1>();
        u1.x = 10;

        for (const auto& up : updates) {
            serialize(s, up);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
