using ArgParse
function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--continue_from_checkpoint", "-c"
        help = "Continue training from selected checkpoint"
        "--metrics", "-m"
        help = "Run test on checkpoint"
    end

    return parse_args(s)
end

println(parse_commandline())
