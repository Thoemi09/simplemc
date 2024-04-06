#include <nlohmann/json.hpp>

#include <iostream>
#include <utility>

struct empty_serializer {
    template <typename T>
    void serialize([[maybe_unused]] const T& t) {
        std::cout << "empty_serializer::serialize" << std::endl;
    }

    template <typename T>
    void deserialize([[maybe_unused]] T& t) const {
        std::cout << "empty_serializer::deserialize" << std::endl;
    }
};

struct json_serializer {
    nlohmann::json j;
    template <typename T>
    void serialize(const T& t) {
        std::cout << "json_serializer::serialize" << std::endl;
        j = t;
        std::cout << j << std::endl;
    }

    template <typename T>
    void deserialize(T& t) const {
        std::cout << "json_serializer::deserialize" << std::endl;
        j.get_to(t);
    }
};

struct mc {
    int steps = 0;

    template <typename S>
    void run(S&& s) {
        for (; steps < 100; ++steps) {
            if (steps % 10 == 0) {
                checkpoint(std::forward<S>(s));
            }
        }
    }

    template <typename S>
    void checkpoint(S&& s) {
        std::forward<S>(s).serialize(*this);
    }
};

void to_json(nlohmann::json& j, const mc& m) {
    j["steps"] = m.steps;
}

int main() {
    mc sim;
    json_serializer js;
    js.j = js.j["mc"];
    sim.run(js);
    std::cout << js.j << std::endl;
}
