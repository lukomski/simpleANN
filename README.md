# Simple Artificial Neural Networks

_Julia lang_

![Julia lang Logo](docs/julia-lang.png "Julia Logo")

---

The repository contains implementations of two simple ANN networks.

#### File structures:

- src/common - common modules for networks. First of all common interface to get datasets, but also calculating metrices.
- src/netjl - implementation of netjl network
- src/netpp - implementation of netpp network

## Run netjl network

#### The easiest way to start training is calling script

```
bash src/netjl/scripts/trainAndPlotAll.bash
```

The script trains netjl network on all implemented datasets with default parameters, calculate metrices and plot results in ouput directory.

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

## Run netpp network

### 1. Install all dependencies

```
julia src/netpp/dependencies.jl
```

### 2. Run main script

```
julia src/netpp/main.jl
```

## Plotter

Plotter is a tool enable calculating metrics and drawing plots. Both metrics and plots are saved to results directory.

### Run Plotter

Example for netpp network:

```
julia src/netjl/plotter/plotter.jl -d 20220803134136 -n netpp
```

#### Display help for Plotter

```
julia src/netjl/plotter/plotter.jl --help
```

---

### Output files can be found at the link:

https://drive.google.com/drive/folders/1QpIz8T4zEhT9N6-xRfVMr2lZZCkgH4XU?usp=sharing
