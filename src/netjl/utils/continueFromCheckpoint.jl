println("Continue from checkpoint")
current_train_folder = parsed_args["continue_from_checkpoint"]
current_checkpoint_folder = "$(default_checkpoint_folder)/$(current_train_folder)"
if (!isdir(current_checkpoint_folder))
    println("Directory not exists: $(current_checkpoint_folder)")
    exit()
end

println("TODO")