using ArgParse

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--directory", "-d"
        help = "Output directory"
        required = true

    end

    return parse_args(s)
end