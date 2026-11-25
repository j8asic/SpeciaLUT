// Built with exceptions disabled: asking for a state that is not in the table
// cannot throw, so it must abort instead of returning some other
// specialization. The program succeeds only if it was aborted.

#include "specialut.hpp"

#include <csignal>
#include <cstdlib>

namespace {

template <int I, int J> int identity() { return 10 * I + J; }

extern "C" void aborted(int /*signal*/) { std::_Exit(0); }

} // namespace

int main() {
  const SpeciaLUT::Chooser<TABULATE(identity), 2, 3> chooser;

  // states inside of the table work as usual
  volatile int first = 1;
  volatile int second = 2;
  if (chooser(first, second) != &identity<1, 2>) {
    return 1;
  }
  if (chooser.find(2, 0) != nullptr) {
    return 2;
  }

  std::signal(SIGABRT, aborted);
  first = 2; // flattens to the slot of (0, 1), but is not a valid state
  second = 0;
  (void)chooser(first, second);
  return 3; // reached only if the lookup returned
}
