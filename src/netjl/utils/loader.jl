#
# load basic paths
#

modules_path = "$(pwd())/src/netjl/modules"
common_path = "$(pwd())/src/common"
datasets_path = "$(pwd())/src/common/datasets"
push!(LOAD_PATH, modules_path)
push!(LOAD_PATH, datasets_path)
push!(LOAD_PATH, common_path)

println("modules_path: $(modules_path)")
println("datasets_path: $(datasets_path)")