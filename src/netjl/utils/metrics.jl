println("Calculate metrics")
current_checkpoint = parsed_args["metrics"]
current_checkpoint_full_path = "$(default_checkpoint_folder)/$(current_checkpoint)"
if (!isfile(current_checkpoint_full_path))
    println("File not exists: $(current_checkpoint_full_path)")
    exit()
end

checkpoint = WeightsModule.loadFromFile(current_checkpoint_full_path)
weights = checkpoint.weights
println("\nTest on file $(current_checkpoint)")
makeTest(test, weights, length(classes))