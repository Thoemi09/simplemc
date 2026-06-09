#include <simplemc/mc.hpp>
#include <simplemc/random/xoshiro256.hpp>
#include <simplemc/serialize/concepts.hpp>
#include <simplemc/utils/simplemc_exception.hpp>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <string_view>
#include <utility>

using namespace simplemc;

namespace {

// Two strongly-typed mock serializers, identical in implementation, distinct in type. Built atop
// nlohmann::json with the same shared-document / json_pointer pattern as json_serializer, so they
// satisfy the simplemc::serializer concept naturally.
//
// The whole point of having two distinct types is to prove that the wrapper + simulation dispatch
// state hooks to one and input-config hooks to the other, with no cross-talk.
// Each instantiation dispatches save_at / load_at through ADL on the wrapped user type first,
// falling back to nlohmann::json's machinery — same pattern as simplemc::json_serializer. This is
// what lets a `simplemc_save(tagged_mock_serializer<Tag>&, const T&)` ADL hook be reached when the
// wrapper or the simulation does `s.save_at(key, value)`.
template <class Tag>
class tagged_mock_serializer {
public:
    explicit tagged_mock_serializer(nlohmann::json doc = {}) :
        root_ { std::make_shared<nlohmann::json>(std::move(doc)) },
        current_ {} { }

    template <class T>
    tagged_mock_serializer save_at(std::string_view key, const T& value) {
        if constexpr (has_simplemc_save<T, tagged_mock_serializer>) {
            auto sub = (*this)[key];
            simplemc_save(sub, value);
        } else {
            (*root_)[current_ / std::string { key }] = value;
        }
        return *this;
    }

    template <class T>
    tagged_mock_serializer load_at(std::string_view key, T& value) const {
        if constexpr (has_simplemc_load<T, tagged_mock_serializer>) {
            const auto sub = (*this)[key];
            simplemc_load(sub, value);
        } else {
            root_->at(current_ / std::string { key }).get_to(value);
        }
        return *this;
    }

    template <class T>
    bool try_load_at(std::string_view key, T& value) const {
        if (!has(key)) return false;
        load_at(key, value);
        return true;
    }

    tagged_mock_serializer operator[](std::string_view key) {
        return tagged_mock_serializer { root_, current_ / std::string { key } };
    }
    tagged_mock_serializer operator[](std::string_view key) const {
        return tagged_mock_serializer { root_, current_ / std::string { key } };
    }

    [[nodiscard]] bool has(std::string_view key) const {
        return root_->contains(current_ / std::string { key });
    }

    nlohmann::json& root() noexcept { return *root_; }
    [[nodiscard]] const nlohmann::json& root() const noexcept { return *root_; }

private:
    tagged_mock_serializer(std::shared_ptr<nlohmann::json> root, nlohmann::json::json_pointer current) :
        root_ { std::move(root) },
        current_ { std::move(current) } { }

    std::shared_ptr<nlohmann::json> root_;
    nlohmann::json::json_pointer current_;
};

struct state_tag { };
struct input_config_tag { };

using mock_state_serializer = tagged_mock_serializer<state_tag>;
using mock_input_config_serializer = tagged_mock_serializer<input_config_tag>;

static_assert(serializer<mock_state_serializer>);
static_assert(serializer<mock_input_config_serializer>);
static_assert(!std::is_same_v<mock_state_serializer, mock_input_config_serializer>);

// User update opting into BOTH flavors, via two ADL hooks typed on different serializer types.
// This proves the wrapper dispatches to the right hook per backend, without any cross-talk.
struct dual_update {
    std::shared_ptr<int> state_counter = std::make_shared<int>(0);
    std::shared_ptr<double> config_threshold = std::make_shared<double>(0.0);
    double attempt() { return 1.0; }
    void accept() { }
};

void simplemc_save(mock_state_serializer& s, const dual_update& u) {
    s.save_at("state_counter", *u.state_counter);
}
void simplemc_load(const mock_state_serializer& s, dual_update& u) {
    s.load_at("state_counter", *u.state_counter);
}
void simplemc_save_input_config(mock_input_config_serializer& s, const dual_update& u) {
    s.save_at("config_threshold", *u.config_threshold);
}
void simplemc_load_input_config(const mock_input_config_serializer& s, dual_update& u) {
    if (s.has("config_threshold")) {
        s.load_at("config_threshold", *u.config_threshold);
    }
}

// User update with ONLY the state hook — the input-config side must silently skip it.
struct state_only_update {
    std::shared_ptr<int> ticks = std::make_shared<int>(0);
    double attempt() { return 1.0; }
    void accept() { }
};

void simplemc_save(mock_state_serializer& s, const state_only_update& u) {
    s.save_at("ticks", *u.ticks);
}
void simplemc_load(const mock_state_serializer& s, state_only_update& u) {
    s.load_at("ticks", *u.ticks);
}

} // namespace

TEST(MCSimulationMixedBackend, InstantiationCompiles) {
    // Smoke test: the three-parameter form instantiates.
    simulation<mock_state_serializer, mock_input_config_serializer> sim;
    dual_update u;
    sim.add_update(u, "tunable", 2.5);
    EXPECT_DOUBLE_EQ(sim.update_data()[0].weight, 2.5);
}

TEST(MCSimulationMixedBackend, StateAndInputConfigDispatchIndependently) {
    // Source simulation with both backends in play.
    simulation<mock_state_serializer, mock_input_config_serializer> src { xoshiro256ss { 7 } };
    dual_update src_u;
    *src_u.state_counter = 42;
    *src_u.config_threshold = 1.25;
    src.add_update(src_u, "tunable", 3.0);
    src.run({ .max_steps = 5, .max_time = 1000.0, .steps_per_cycle = 1, .cycles_per_check = 5 });
    src.accumulate_stats();

    // Save state to the state serializer; save input config to the (different-type) input-config
    // serializer.
    mock_state_serializer state_w;
    state_w.save_at("sim", src);

    mock_input_config_serializer input_w;
    auto input_sim = input_w["sim"];
    simplemc_save_input_config(input_sim, src);

    // State serializer should have the state-only fields:
    EXPECT_TRUE(state_w.root()["sim"].contains("rng"));
    EXPECT_TRUE(state_w.root()["sim"].contains("cumulative_steps"));
    EXPECT_TRUE(state_w.root()["sim"]["updates"]["tunable"].contains("cumulative_nprops"));
    EXPECT_TRUE(state_w.root()["sim"]["updates"]["tunable"]["user"].contains("state_counter"));

    // Input-config serializer should have only the input-config fields, no state fields:
    EXPECT_FALSE(input_w.root()["sim"].contains("rng"));
    EXPECT_FALSE(input_w.root()["sim"].contains("cumulative_steps"));
    EXPECT_FALSE(input_w.root()["sim"]["updates"]["tunable"].contains("cumulative_nprops"));
    EXPECT_TRUE(input_w.root()["sim"]["updates"]["tunable"].contains("weight"));
    EXPECT_TRUE(input_w.root()["sim"]["updates"]["tunable"]["user"].contains("config_threshold"));

    // Destination: load state from state serializer, then input-config from input-config
    // serializer. Cross-talk would show up as wrong values or extra throws.
    simulation<mock_state_serializer, mock_input_config_serializer> dst;
    dual_update dst_u;
    auto dst_state_counter = dst_u.state_counter;
    auto dst_config_threshold = dst_u.config_threshold;
    dst.add_update(dst_u, "tunable", 1.0);

    const mock_state_serializer state_r { state_w.root() };
    state_r.load_at("sim", dst);

    const mock_input_config_serializer input_r { input_w.root() };
    const auto input_sim_r = input_r["sim"];
    simplemc_load_input_config(input_sim_r, dst);

    // State round-trip:
    EXPECT_EQ(dst.stats().cumulative_steps, src.stats().cumulative_steps);
    EXPECT_EQ(*dst_state_counter, 42);

    // Input-config round-trip:
    EXPECT_DOUBLE_EQ(dst.update_data()[0].weight, 3.0);
    EXPECT_DOUBLE_EQ(*dst_config_threshold, 1.25);
}

TEST(MCSimulationMixedBackend, OneSidedOptInSilentlySkipsOtherFlavor) {
    // state_only_update has only the state hook. Input-config save/load should be no-ops on the
    // user slot; no exceptions, no fields written.
    simulation<mock_state_serializer, mock_input_config_serializer> sim;
    state_only_update u;
    *u.ticks = 99;
    sim.add_update(u, "ticker", 1.0);

    mock_state_serializer state_w;
    state_w.save_at("sim", sim);
    EXPECT_TRUE(state_w.root()["sim"]["updates"]["ticker"]["user"].contains("ticks"));

    mock_input_config_serializer input_w;
    auto input_sim = input_w["sim"];
    EXPECT_NO_THROW(simplemc_save_input_config(input_sim, sim));
    // Per-entry "weight" is still written (it's on update_stats, not the user payload), but the
    // user slot is empty because state_only_update doesn't opt into input-config.
    const auto& ticker = input_w.root()["sim"]["updates"]["ticker"];
    EXPECT_TRUE(ticker.contains("weight"));
    EXPECT_FALSE(ticker.contains("user"));
}
