# Simple Artificial Neural Networks

_Julia lang_

![Julia lang Logo](docs/julia-lang.png "Julia Logo")

---

## Run netjl network

#### Start training

```
julia src/netjl/main.jl
```

or alternatively open the main file and click Run button for Julia REPL

#### Show metrics of checkpoint

```
julia src/netjl/main.jl -m 20220824215915/weight.0
```

#### Display help

```
julia src/netjl/main.jl --help
```

---

### Plotter

Plotter is a tool enable calculating metrics and drawing plots. Both metrics and plots are saved to results directory.

#### Run Plotter

```
julia src/netjl/plotter/plotter.jl -d 20220903134136
```

#### Display help for Plotter

```
julia src/netjl/plotter/plotter.jl --help
```

---

## Run netpp network

### 1. Go to src/netpp directory

```
cd src/netpp
```

### 2. Run main script

```
julia main.jl
```
