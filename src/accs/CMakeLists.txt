# Build accs library for simplemc.
# 
# The following targets are defined:
# ---------------------------------
# simplemc-accs (simplemc::simplemc-accs)

add_library(simplemc-accs INTERFACE)
add_library(simplemc::simplemc-accs ALIAS simplemc-accs)
# target_sources(simplemc-accs
#     PRIVATE
#         acc_traits.cpp
#         mean_acc.cpp
#         var_acc.cpp
#         blockvar_acc.cpp
#         autocorr_acc.cpp
#         batch_acc.cpp
#         jackknife.cpp
#         acc_json.cpp
#         acc_mpi.cpp)
target_link_libraries(simplemc-accs
    INTERFACE
        simplemc::simplemc-numeric
        simplemc::simplemc-mpi)
