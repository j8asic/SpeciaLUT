// SpeciaLUT tests: compile-time checks where possible, run-time checks for the
// rest. No test framework needed; the program returns the number of failures.

#include "specialut.hpp"

#include <array>
#include <concepts>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace {

using SpeciaLUT::arg_t;
using SpeciaLUT::Chooser;
using SpeciaLUT::values;

int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      ++failures;                                                              \
      std::printf("%s:%d: failed: %s\n", __FILE__, __LINE__, #condition);      \
    }                                                                          \
  } while (false)

#define CHECK_OUT_OF_RANGE(expression)                                         \
  do {                                                                         \
    bool thrown = false;                                                       \
    try {                                                                      \
      (void)(expression);                                                      \
    } catch (std::out_of_range const &) {                                      \
      thrown = true;                                                           \
    }                                                                          \
    if (!thrown) {                                                             \
      ++failures;                                                              \
      std::printf("%s:%d: did not throw: %s\n", __FILE__, __LINE__,            \
                  #expression);                                                \
    }                                                                          \
  } while (false)

/// Can the table C be asked for a specialization with keys of these types?
template <typename C, typename... K>
concept can_choose = requires(K... keys) { C::at(keys...); };

// -----------------------------------------------------------------------------
// Free function, parameters tabulated by their number of states
// -----------------------------------------------------------------------------

template <bool A, int B, unsigned C> constexpr int counted(int x) {
  return 1000 * int(A) + 100 * B + 10 * int(C) + x;
}

using Counted = Chooser<TABULATE(counted), 2, 3, 4>;

static_assert(Counted::rank == 3);
static_assert(Counted::size() == 24);
static_assert(Counted::extents == std::array<arg_t, 3>{2, 3, 4});
static_assert(std::is_same_v<Counted::pointer, int (*)(int)>);
static_assert(std::is_empty_v<Counted>);

// the table holds exactly the specialization that was asked for
static_assert(Counted::at(0, 0, 0) == &counted<false, 0, 0>);
static_assert(Counted::at(1, 2, 3) == &counted<true, 2, 3>);
static_assert(Counted::at(true, 1, 2u) == &counted<true, 1, 2>);
static_assert(Counted::at(1, 2, 3)(4) == 1234);
static_assert(Counted::find(1, 0, 3) == &counted<true, 0, 3>);

// every parameter is checked against its own range: (2, 0, 0) flattens to the
// same table slot as (0, 1, 0), and must not be mistaken for it
static_assert(Counted::find(2, 0, 0) == nullptr);
static_assert(Counted::find(0, 3, 0) == nullptr);
static_assert(Counted::find(0, 0, 4) == nullptr);
static_assert(Counted::find(0, -1, 0) == nullptr);

// the number of keys and their types are part of the interface
static_assert(can_choose<Counted, int, int, int>);
static_assert(can_choose<Counted, bool, long, unsigned char>);
static_assert(!can_choose<Counted, int, int>);
static_assert(!can_choose<Counted, int, int, int, int>);
static_assert(!can_choose<Counted, int, const char *, int>);

// an object of the chooser works in constant expressions too
constexpr Counted counted_chooser;
static_assert(counted_chooser(1, 2, 3) == &counted<true, 2, 3>);

#ifdef SPECIALUT_HAS_STATIC_CALL_OP
static_assert(Counted::operator()(1, 2, 3) == &counted<true, 2, 3>);
#endif

void test_counted() {
  const Counted chooser;
  for (int a = -1; a <= 3; ++a) {
    for (int b = -1; b <= 4; ++b) {
      for (int c = -1; c <= 5; ++c) {
        const bool valid =
            a >= 0 && a < 2 && b >= 0 && b < 3 && c >= 0 && c < 4;
        if (valid) {
          CHECK(chooser(a, b, c)(7) == 1000 * a + 100 * b + 10 * c + 7);
          CHECK(Counted::find(a, b, c) == chooser(a, b, c));
        } else {
          CHECK(Counted::find(a, b, c) == nullptr);
          CHECK_OUT_OF_RANGE(chooser(a, b, c));
          CHECK_OUT_OF_RANGE(Counted::at(a, b, c));
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Keys of an unscoped enum, which converts to an index by itself
// -----------------------------------------------------------------------------

enum Legacy { legacy_a, legacy_b, legacy_c };

template <int Mode> constexpr int single(int x) { return 10 * Mode + x; }

using Single = Chooser<TABULATE(single), 3>;

static_assert(can_choose<Single, Legacy>);
static_assert(Single::at(legacy_c) == &single<2>);

void test_unscoped_enum_key() {
  const Single chooser;
  Legacy mode = legacy_b; // an lvalue, as the first and only key
  CHECK(chooser(mode)(5) == 15);
  mode = legacy_c;
  CHECK(chooser(mode)(5) == 25);
}

// -----------------------------------------------------------------------------
// Enum parameters, tabulated by an enumerator that counts the states
// -----------------------------------------------------------------------------

enum class Kernel { cubic, quintic, wendland, count };

template <Kernel K, bool Flag> constexpr int by_enum(int x) {
  return 100 * int(K) + 10 * int(Flag) + x;
}

using ByEnum = Chooser<TABULATE(by_enum), Kernel::count, 2>;

static_assert(ByEnum::size() == 6);
static_assert(ByEnum::extents == std::array<arg_t, 2>{3, 2});
static_assert(ByEnum::at(Kernel::cubic, false) ==
              &by_enum<Kernel::cubic, false>);
static_assert(ByEnum::at(Kernel::wendland, true) ==
              &by_enum<Kernel::wendland, true>);
static_assert(ByEnum::find(Kernel::count, true) == nullptr);
static_assert(ByEnum::find(static_cast<Kernel>(-1), false) == nullptr);
static_assert(ByEnum::find(Kernel::cubic, 2) == nullptr);

// the enum parameter is chosen by the enum, not by a bare number, and a scoped
// enum is not accepted where a number is expected
static_assert(can_choose<ByEnum, Kernel, bool>);
static_assert(!can_choose<ByEnum, int, bool>);
static_assert(!can_choose<ByEnum, Kernel, Kernel>);
static_assert(!can_choose<Counted, Kernel, int, int>);

void test_enum() {
  const ByEnum chooser;
  for (int k = 0; k < 3; ++k) {
    for (int flag = 0; flag < 2; ++flag) {
      CHECK(chooser(static_cast<Kernel>(k), flag)(1) ==
            100 * k + 10 * flag + 1);
    }
  }
  CHECK_OUT_OF_RANGE(chooser(Kernel::count, false));
  CHECK_OUT_OF_RANGE(chooser(static_cast<Kernel>(7), true));
}

// -----------------------------------------------------------------------------
// Parameters tabulated by an explicit list of values
// -----------------------------------------------------------------------------

template <int Dim, Kernel K> constexpr int by_value(int x) {
  return 100 * Dim + 10 * int(K) + x;
}

using ByValue = Chooser<TABULATE(by_value), values<2, 3>,
                        values<Kernel::cubic, Kernel::wendland>>;

static_assert(ByValue::size() == 4);
static_assert(ByValue::at(2, Kernel::cubic) == &by_value<2, Kernel::cubic>);
static_assert(ByValue::at(3, Kernel::wendland) ==
              &by_value<3, Kernel::wendland>);
static_assert(ByValue::at(std::size_t{3}, Kernel::cubic) ==
              &by_value<3, Kernel::cubic>);
static_assert(ByValue::find(0, Kernel::cubic) == nullptr);
static_assert(ByValue::find(4, Kernel::cubic) == nullptr);
static_assert(ByValue::find(2, Kernel::quintic) == nullptr); // not listed
static_assert(!can_choose<ByValue, int, int>);

// the listed order does not have to be ascending, nor contiguous
template <int Width> constexpr int widths(int x) { return Width * x; }
using Widths = Chooser<TABULATE(widths), values<16, 4, 8>>;
static_assert(Widths::at(4) == &widths<4>);
static_assert(Widths::at(8) == &widths<8>);
static_assert(Widths::at(16) == &widths<16>);
static_assert(Widths::find(2) == nullptr);

// all three ways of describing a parameter, mixed
template <bool A, Kernel K, int Dim> constexpr int mixed() {
  return 1000 * int(A) + 100 * int(K) + Dim;
}
using Mixed = Chooser<TABULATE(mixed), 2, Kernel::count, values<2, 3>>;
static_assert(Mixed::size() == 12);
static_assert(Mixed::at(true, Kernel::quintic, 3) ==
              &mixed<true, Kernel::quintic, 3>);
static_assert(Mixed::at(true, Kernel::quintic, 3)() == 1103);

int generic_width(int width, int x) { return width * x; }

void test_values() {
  const ByValue chooser;
  for (int dim = 0; dim <= 5; ++dim) {
    for (int k = 0; k <= 3; ++k) {
      const auto kernel = static_cast<Kernel>(k);
      if ((dim == 2 || dim == 3) && (k == 0 || k == 2)) {
        CHECK(chooser(dim, kernel)(1) == 100 * dim + 10 * k + 1);
      } else {
        CHECK(ByValue::find(dim, kernel) == nullptr);
        CHECK_OUT_OF_RANGE(chooser(dim, kernel));
      }
    }
  }

  // specialize the common cases, fall back to generic code for the others
  for (int width = 1; width <= 20; ++width) {
    const auto specialized = Widths::find(width);
    const int result = specialized ? specialized(3) : generic_width(width, 3);
    CHECK(result == 3 * width);
    CHECK((specialized != nullptr) ==
          (width == 4 || width == 8 || width == 16));
  }
}

// -----------------------------------------------------------------------------
// Member functions
// -----------------------------------------------------------------------------

struct Body {
  long base = 0;

  template <int I> long add(int x) { return base += I + x; }
  template <int I> long peek(int x) const { return base + 10 * I + x; }
  template <int I> long &ref() { return base; }
  template <bool Negate> static long sign(long x) { return Negate ? -x : x; }

  long from_within(int i, int x) {
    constexpr auto adders = TABULATE(Body::add);
    const Chooser<adders, 3> chooser;
    return chooser(this, i)(x);
  }
};

struct DerivedBody : Body {
  long extra = 5;
};

using Adders = Chooser<TABULATE(Body::add), 3>;
using Peekers = Chooser<TABULATE(Body::peek), 3>;
using Refs = Chooser<TABULATE(Body::ref), 2>;
using Signs = Chooser<TABULATE(Body::sign), 2>;

static_assert(std::is_same_v<Adders::pointer, long (Body::*)(int)>);
static_assert(std::is_same_v<Peekers::pointer, long (Body::*)(int) const>);
static_assert(std::is_same_v<Signs::pointer, long (*)(long)>);
static_assert(Adders::at(2) == &Body::add<2>);
static_assert(Adders::find(3) == nullptr);

// an object can only be bound to a table of member functions of its class
static_assert(std::invocable<const Adders &, Body &, int>);
static_assert(std::invocable<const Adders &, Body *, int>);
static_assert(std::invocable<const Adders &, DerivedBody &, int>);
static_assert(std::invocable<const Peekers &, const Body &, int>);
static_assert(!std::invocable<const Adders &, Kernel &, int>);
static_assert(!std::invocable<const Adders &, Body &, int, int>);
static_assert(!std::invocable<const Counted &, Body &, int, int, int>);
static_assert(!std::invocable<const Signs &, Body &, bool>);

// the bound call returns what the member function returns, references too
static_assert(std::is_same_v<decltype(std::declval<const Refs &>()(
                                 std::declval<Body &>(), 0)()),
                             long &>);
static_assert(std::is_same_v<decltype(std::declval<const Adders &>()(
                                 std::declval<Body &>(), 0)(1)),
                             long>);

void test_members() {
  const Adders adders;
  Body body{100};

  CHECK((body.*adders(1))(10) == 111); // plain member pointer
  CHECK(adders(body, 2)(10) == 123);   // bound to a reference
  CHECK(adders(&body, 0)(10) == 133);  // bound to a pointer
  CHECK(body.base == 133);
  CHECK(body.from_within(2, 1) == 136);
  CHECK_OUT_OF_RANGE(adders(body, 3));
  CHECK_OUT_OF_RANGE(adders(&body, -1));

  const Peekers peekers;
  const Body constant{7};
  CHECK(peekers(constant, 2)(1) == 28);
  CHECK(peekers(&constant, 1)(1) == 18);

  DerivedBody derived;
  derived.base = 1;
  CHECK(adders(derived, 1)(1) == 3);
  CHECK(derived.base == 3);

  const Refs refs;
  refs(body, 1)() = 42; // writes through the returned reference
  CHECK(body.base == 42);

  const Signs signs;
  CHECK(signs(true)(5) == -5);
  CHECK(signs(false)(5) == 5);
}

// -----------------------------------------------------------------------------
// Functors and lambdas
// -----------------------------------------------------------------------------

struct Functor {
  int offset = 0;
  template <bool Yes> int operator()(int n) const {
    return Yes ? offset + n : offset - n;
  }
};

void test_functor() {
  const Chooser<TABULATE_FUNCTOR(Functor), 2> chooser;
  const Functor functor{100};
  CHECK((functor.*chooser(true))(1) == 101);
  CHECK(chooser(functor, false)(1) == 99);
}

template <Kernel K, int Dim> constexpr int direct(int x) {
  return 100 * int(K) + 10 * Dim + x;
}

/// A lambda that is gone by the time its chosen specialization is called
auto chosen_from_temporary(bool negate, int scale) {
  return SpeciaLUT::choose_lambda<2, 3>(
      [base = 1000]<bool Negate, int Scale>(int x) {
        return base + (Negate ? -1 : 1) * (Scale + 1) * x;
      },
      negate, scale);
}

void test_lambdas() {
  int captured = 2;
  auto lambda = [&]<bool T>(int x) { return int(T) + x * captured; };

  const Chooser<TABULATE_LAMBDA(lambda), 2> chooser;
  CHECK((lambda.*chooser(true))(10) == 21);
  CHECK(chooser(lambda, false)(10) == 20);

  // choose_lambda: no macro, no table object
  CHECK(SpeciaLUT::choose_lambda<2>(lambda, true)(10) == 21);
  captured = 3;
  CHECK(SpeciaLUT::choose_lambda<2>(lambda, false)(10) == 30);

  // an lvalue lambda is held by reference, so its state is shared
  auto counter = [total = 0]<int Step>() mutable { return total += Step; };
  CHECK(SpeciaLUT::choose_lambda<4>(counter, 3)() == 3);
  CHECK(SpeciaLUT::choose_lambda<4>(counter, 2)() == 5);
  const auto bound = SpeciaLUT::choose_lambda<4>(counter, 1);
  CHECK(bound() == 6);
  CHECK(bound() == 7);

  // a temporary lambda is moved into the result, and stays alive in it
  auto kept = chosen_from_temporary(true, 2);
  CHECK(kept(1) == 997);
  CHECK(chosen_from_temporary(false, 0)(5) == 1005);
  CHECK_OUT_OF_RANGE(chosen_from_temporary(false, 3));

  // the way to dispatch to a function without naming a table for it
  for (int k = 0; k < 3; ++k) {
    for (int dim = 2; dim <= 3; ++dim) {
      const auto kernel = static_cast<Kernel>(k);
      const int result = SpeciaLUT::choose_lambda<Kernel::count, values<2, 3>>(
          []<Kernel K, int Dim>(int x) { return direct<K, Dim>(x); }, kernel,
          dim)(4);
      CHECK(result == 100 * k + 10 * dim + 4);
    }
  }
}

// -----------------------------------------------------------------------------
// Function types
// -----------------------------------------------------------------------------

template <int I> constexpr int never_throws(int x) noexcept { return I + x; }
using NeverThrows = Chooser<TABULATE(never_throws), 2>;
static_assert(std::is_same_v<NeverThrows::pointer, int (*)(int) noexcept>);
static_assert(noexcept(std::declval<NeverThrows::pointer>()(1)));

template <int I> void by_reference(int &out, const int &in) { out = in + I; }

void test_arguments() {
  const Chooser<TABULATE(by_reference), 3> chooser;
  int out = 0;
  chooser(2)(out, 40);
  CHECK(out == 42);
}

// -----------------------------------------------------------------------------
// States: the parameters described once, and chosen once, for several tables
// -----------------------------------------------------------------------------

using Options = SpeciaLUT::States<Kernel::count, values<2, 3>, 2>;

template <Kernel K, int Dim, bool Flag> constexpr int density(int x) {
  return 1000 + 100 * int(K) + 10 * Dim + int(Flag) + x;
}

template <Kernel K, int Dim, bool Flag> constexpr long forces(long a, long b) {
  return (100 * int(K) + 10 * Dim + int(Flag)) * a + b;
}

using Density = Options::Chooser<TABULATE(density)>;
using Forces = Options::Chooser<TABULATE(forces)>;

static_assert(Options::rank == 3);
static_assert(Options::size() == 12);
static_assert(Options::extents == std::array<arg_t, 3>{3, 2, 2});
static_assert(std::is_same_v<Density::state_type, Options>);
static_assert(std::is_same_v<Forces::state_type, Options>);
static_assert(std::is_same_v<Counted::state_type, SpeciaLUT::States<2, 3, 4>>);
static_assert(Density::rank == 3 && Density::size() == 12);

// states are checked when they are made, by the types of the parameters too
static_assert(Options::contains(Kernel::wendland, 3, true));
static_assert(!Options::contains(Kernel::count, 3, true));
static_assert(!Options::contains(Kernel::cubic, 4, true));
static_assert(!Options::contains(Kernel::cubic, 2, 2));
static_assert(std::constructible_from<Options, Kernel, int, bool>);
static_assert(!std::constructible_from<Options, int, int, bool>);
static_assert(!std::constructible_from<Options, Kernel, int>);
static_assert(!std::is_default_constructible_v<Options>);
static_assert(!std::is_convertible_v<int, SpeciaLUT::States<3>>);

constexpr Options quintic_3d(Kernel::quintic, 3, true);
static_assert(quintic_3d.index() == 1 + 3 * (1 + 2 * 1));
static_assert(quintic_3d == Options(Kernel::quintic, 3, true));
static_assert(quintic_3d != Options(Kernel::quintic, 2, true));

constexpr Density density_chooser;
constexpr Forces forces_chooser;
static_assert(density_chooser[quintic_3d] ==
              &density<Kernel::quintic, 3, true>);
static_assert(forces_chooser[quintic_3d] == &forces<Kernel::quintic, 3, true>);
static_assert(Density::find(quintic_3d) ==
              Density::at(Kernel::quintic, 3, true));

void test_states() {
  const Density densities;
  const Forces all_forces;
  for (int k = 0; k < 3; ++k) {
    for (int dim = 2; dim <= 3; ++dim) {
      for (int flag = 0; flag < 2; ++flag) {
        const auto kernel = static_cast<Kernel>(k);
        const Options options(kernel, dim, flag != 0); // checked here, once
        const int id = 100 * k + 10 * dim + flag;
        CHECK(densities[options](1) == 1000 + id + 1);
        CHECK(all_forces[options](2, 3) == 2 * id + 3);
        CHECK(densities[options] == densities(kernel, dim, flag != 0));
      }
    }
  }
  CHECK_OUT_OF_RANGE(Options(Kernel::count, 2, false));
  CHECK_OUT_OF_RANGE(Options(Kernel::cubic, 5, false));
  CHECK_OUT_OF_RANGE(Options(Kernel::cubic, 2, 7));

  // a single parameter, and states that are assigned later
  using Mode = SpeciaLUT::States<3>;
  const Mode::Chooser<TABULATE(single)> singles;
  Mode mode(2);
  CHECK(singles[mode](1) == 21);
  mode = Mode(0);
  CHECK(singles[mode](1) == 1);

  // member functions, bound to an object
  const Mode::Chooser<TABULATE(Body::add)> adders;
  Body body{100};
  CHECK(adders(body, Mode(2))(10) == 112);
  CHECK(adders(&body, Mode(1))(10) == 123);
  CHECK((body.*adders[Mode(0)])(1) == 124);
}

// -----------------------------------------------------------------------------
// Combinations that the template rejects are left out of the table
// -----------------------------------------------------------------------------

// there is no such thing as a viscous quintic kernel
template <Kernel K, bool Viscous>
  requires(!(K == Kernel::quintic && Viscous))
constexpr int constrained(int x) {
  return 100 * int(K) + 10 * int(Viscous) + x;
}

using Constrained = Chooser<TABULATE(constrained), Kernel::count, 2>;

// 6 combinations of the states, 5 specializations
static_assert(Constrained::state_type::size() == 6);
static_assert(Constrained::size() == 5);
static_assert(Constrained::find(Kernel::quintic, true) == nullptr);
static_assert(Constrained::find(Kernel::quintic, false) ==
              &constrained<Kernel::quintic, false>);
static_assert(Constrained::at(Kernel::wendland, true) ==
              &constrained<Kernel::wendland, true>);
static_assert(Constrained::find(Constrained::state_type(Kernel::quintic,
                                                        true)) == nullptr);

// a table without rejected combinations has as many of both
static_assert(Counted::size() == Counted::state_type::size());

// the pointer type comes from the first specialization that does exist
template <bool A, int B>
  requires(A || B == 2)
constexpr int late(int x) {
  return 10 * int(A) + B + x;
}

using Late = Chooser<TABULATE(late), 2, 3>;

static_assert(std::is_same_v<Late::pointer, int (*)(int)>);
static_assert(Late::size() == 4);
static_assert(Late::find(0, 0) == nullptr);
static_assert(Late::find(0, 1) == nullptr);
static_assert(Late::find(1, 0) == &late<true, 0>);
static_assert(Late::find(0, 2) == &late<false, 2>);

struct Picky {
  template <int I>
    requires(I != 1)
  int get() const {
    return I;
  }
};

using Pickies = Chooser<TABULATE(Picky::get), 3>;

static_assert(Pickies::size() == 2);
static_assert(Pickies::find(1) == nullptr);

void test_rejected_combinations() {
  const Constrained chooser;
  CHECK(chooser(Kernel::cubic, true)(1) == 11);
  CHECK(chooser(Kernel::quintic, false)(1) == 101);

  // asking for a rejected combination is refused like a state out of range
  CHECK(Constrained::find(Kernel::quintic, true) == nullptr);
  CHECK_OUT_OF_RANGE(chooser(Kernel::quintic, true));
  CHECK_OUT_OF_RANGE(Constrained::at(Kernel::quintic, true));

  // the states are valid ones, only this function has nothing for them
  const Constrained::state_type states(Kernel::quintic, true);
  CHECK(Constrained::find(states) == nullptr);
  CHECK_OUT_OF_RANGE(chooser[states]);

  const Pickies pickies;
  const Picky picky;
  CHECK(pickies(picky, 2)() == 2);
  CHECK_OUT_OF_RANGE(pickies(picky, 1));
  CHECK_OUT_OF_RANGE(pickies(&picky, Pickies::state_type(1)));

  auto lambda = [&]<int I>
    requires(I != 0)
  (int x) { return I * x; };
  CHECK(SpeciaLUT::choose_lambda<3>(lambda, 2)(5) == 10);
  CHECK_OUT_OF_RANGE(SpeciaLUT::choose_lambda<3>(lambda, 0));
}

// the short macro names are plain aliases of the prefixed ones
static_assert(Chooser<SPECIALUT_TABULATE(single), 3>::at(1) == &single<1>);

} // namespace

int main() {
  test_counted();
  test_unscoped_enum_key();
  test_enum();
  test_values();
  test_members();
  test_functor();
  test_lambdas();
  test_arguments();
  test_states();
  test_rejected_combinations();

  if (failures == 0) {
    std::puts("All SpeciaLUT tests passed.");
  } else {
    std::printf("%d check(s) failed.\n", failures);
  }
  return failures == 0 ? 0 : 1;
}
