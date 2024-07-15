SET(UTILITIES_CU_SOURCES
  ErrorHandling/HandleUnsuccessfulCUDACall.cu
  )

ADD_LIBRARY(Utilities
  ${UTILITIES_CU_SOURCES}
  )