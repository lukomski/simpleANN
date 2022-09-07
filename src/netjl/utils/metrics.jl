println("Calculate metrics")
current_checkpoint_path = parsed_args["metrics"]
if (!isfile(current_checkpoint_path))
    println("File not exists: $(current_checkpoint_path)")
    exit()
end

checkpoint = WeightsModule.loadFromFile(current_checkpoint_path)
weights = checkpoint.weights
println("\nTest on file $(current_checkpoint_path)")
makeTest(test, weights, length(classes))