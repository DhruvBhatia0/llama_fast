//#include <boost/test/unit_test.hpp> // Wrong.
// https://stackoverflow.com/questions/33644088/linker-error-while-building-unit-tests-with-boost
//#include <boost/test/included/unit_test.hpp> // This works.
#include "gtest/gtest.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <torch/extension.h>

namespace GoogleUnitTests
{
namespace Operations
{

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
TEST(RMSNormTests, DefaultConstructible)
{

  EXPECT_TRUE(true);
}

} // namespace Operations
} // namespace GoogleUnitTests

/*
BOOST_AUTO_TEST_SUITE(Operations)
BOOST_AUTO_TEST_SUITE(RMSNorm_tests)

//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(test_case)
{
}

BOOST_AUTO_TEST_SUITE_END() // RMSNorm_tests
BOOST_AUTO_TEST_SUITE_END() // Operations
*/