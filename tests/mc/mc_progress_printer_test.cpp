// SPDX-FileCopyrightText: 2026 Thomas Hahn
// SPDX-License-Identifier: MIT

#include <simplemc/mc.hpp>
#include <simplemc/mc/progress_printer.hpp>
#include <simplemc/utils/file_io.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdio>

using namespace simplemc;

namespace {

// Read all of `fp` (rewinding first) into a std::string for diagnostic assertions.
std::string read_tmpfile(std::FILE* fp) {
    std::fflush(fp);
    std::fseek(fp, 0, SEEK_SET);
    std::string out;
    std::array<char, 256> buf {};
    while (auto n = std::fread(buf.data(), 1, buf.size(), fp)) {
        out.append(buf.data(), n);
    }
    return out;
}

} // namespace

// Test that a progress printer prints on the first call.
TEST(MCProgressPrinter, PrintsOnFirstCall) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "test", .max_steps = 100 }, false, true, 0.01, fh };
    simulation_ctx x { .steps_done = 42 };
    pp(x);
    const auto text = read_tmpfile(fh);
    EXPECT_NE(text.find("test "), std::string::npos);
    EXPECT_NE(text.find("42/100"), std::string::npos);
}

// Test the constructor that takes only budgets and builds the progress_line internally.
TEST(MCProgressPrinter, BudgetCtor) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { 100, 0.0, false, true, 0.01, fh };
    pp(simulation_ctx { .steps_done = 25 });
    const auto text = read_tmpfile(fh);
    EXPECT_NE(text.find("steps 25/100"), std::string::npos);
    EXPECT_NE(text.find("25.0%"), std::string::npos);
}

// Test that a progress printer throttles subsequent calls until enough progress has been made.
TEST(MCProgressPrinter, ThrottlesSubsequentCalls) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "tp", .max_steps = 100 }, false, true, 0.5, fh };
    pp(simulation_ctx { .steps_done = 1 }); // prints (first call)
    pp(simulation_ctx { .steps_done = 2 }); // suppressed
    pp(simulation_ctx { .steps_done = 3 }); // suppressed
    const auto text = read_tmpfile(fh);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
}

TEST(MCProgressPrinter, EmitsAfterEnoughProgress) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "ep", .max_steps = 100 }, false, true, 0.25, fh };
    pp(simulation_ctx { .steps_done = 0 }); // prints (first call)
    pp(simulation_ctx { .steps_done = 10 }); // +10% < 25% -> suppressed
    pp(simulation_ctx { .steps_done = 20 }); // +20% < 25% -> suppressed
    pp(simulation_ctx { .steps_done = 30 }); // +30% >= 25% -> prints
    const auto text = read_tmpfile(fh);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 2);
}

// Test disabling the progress printer in the constructor.
TEST(MCProgressPrinter, DisabledSuppressesOutput) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "rg", .max_steps = 100 }, false, /*enabled=*/false, 0.01, fh };
    pp(simulation_ctx { .steps_done = 1 });
    pp(simulation_ctx { .steps_done = 2 });
    const auto text = read_tmpfile(fh);
    EXPECT_TRUE(text.empty());
}

// Test the basic progress bar.
TEST(MCProgressPrinter, ShowsBarWhenEnabled) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "bar", .max_steps = 100, .show_bar = true, .bar_width = 10 }, false, true, 0.01,
        fh };
    pp(simulation_ctx { .steps_done = 50 });
    const auto text = read_tmpfile(fh);
    EXPECT_NE(text.find("[#####-----]"), std::string::npos);
    EXPECT_NE(text.find("50.0%"), std::string::npos);
}

// Test a custom progress bar.
TEST(MCProgressPrinter, ProgressLineCustomGlyphsAndRender) {
    const progress_line line {
        .prefix = "g", .max_steps = 100, .show_bar = true, .bar_width = 10, .fill_char = '*', .empty_char = '.'
    };
    const auto text = line.render(simulation_ctx { .steps_done = 30 });
    EXPECT_NE(text.find("[***.......]"), std::string::npos);
    EXPECT_NE(text.find("steps 30/100"), std::string::npos);
    EXPECT_NE(text.find("30.0%"), std::string::npos);
    EXPECT_EQ(text.find('\n'), std::string::npos); // render() adds no terminator
}

// Test that the progress bar uses the max. of the the budgets (steps vs runtime).
TEST(MCProgressPrinter, BarUsesMaxOfBudgets) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "b", .max_steps = 100, .max_time = 100.0, .show_bar = true, .bar_width = 10 },
        false, true, 0.01, fh };
    pp(simulation_ctx { .steps_done = 20 });
    const auto text = read_tmpfile(fh);
    EXPECT_NE(text.find("[##--------]"), std::string::npos);
    EXPECT_NE(text.find("20.0%"), std::string::npos);
}

// Test the in-place progress line.
TEST(MCProgressPrinter, InplaceUsesCarriageReturn) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "ip", .max_steps = 100 }, /*in_place=*/true, true, 0.01, fh };
    pp(simulation_ctx { .steps_done = 10 });
    pp(simulation_ctx { .steps_done = 20 });
    const auto text = read_tmpfile(fh);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 0);
    EXPECT_GT(std::count(text.begin(), text.end(), '\r'), 0);
}

// Test that the progress bar is suppressed in case of no given budgets.
TEST(MCProgressPrinter, NoBarWithoutBudget) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "nb", .show_bar = true }, false, true, 0.01, fh };
    pp(simulation_ctx { .steps_done = 7 });
    const auto text = read_tmpfile(fh);
    EXPECT_EQ(text.find('#'), std::string::npos);
    EXPECT_NE(text.find("steps 7"), std::string::npos);
}

// Test that the 100% line is always printed.
TEST(MCProgressPrinter, AlwaysEmitsCompletion) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "c", .max_steps = 100 }, false, true, 0.5, fh };
    pp(simulation_ctx { .steps_done = 90 }); // first call prints (90%)
    pp(simulation_ctx { .steps_done = 100 }); // +10% < 50% but complete -> prints
    const auto text = read_tmpfile(fh);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 2);
    EXPECT_NE(text.find("100.0%"), std::string::npos);
}

// Test that no more lines are printer after 100%.
TEST(MCProgressPrinter, StopsAfterCompletion) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "s", .max_steps = 100 }, false, true, 0.01, fh };
    pp(simulation_ctx { .steps_done = 100 }); // completes -> prints
    pp(simulation_ctx { .steps_done = 100 }); // already finished -> no-op
    pp(simulation_ctx { .steps_done = 120 }); // already finished -> no-op
    const auto text = read_tmpfile(fh);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
}

// Test that an in-place progress printer commits the final line with a newline.
TEST(MCProgressPrinter, InplaceCompletionCommitsWithNewline) {
    const file_handle fh { std::tmpfile() };
    ASSERT_NE(fh.get(), nullptr);
    progress_printer pp { { .prefix = "ic", .max_steps = 100 }, /*in_place=*/true, true, 0.01, fh };
    pp(simulation_ctx { .steps_done = 50 }); // live redraw -> '\r'
    pp(simulation_ctx { .steps_done = 100 }); // completion -> committed with '\n'
    const auto text = read_tmpfile(fh);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\r'), 1);
    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), 1);
}
