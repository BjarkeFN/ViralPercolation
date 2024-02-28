# ViralPercolation
Cellular automaton simulations of viral spread in tissue, including IFN dynamics.


Compile (on UNIX-like systems equipped with GNU C++ Compiler) by running the shell script compile_sim.sh:

```chmod +x compile_sim.sh```

```./compile_sim.sh NOVAa_sim.cpp```


Then, simulations can be run with

```./NOVAa {L} {T} {p_a} {R}```

(Replacing {L} {T} {p_a} {R} with the desired parameter values. For a test run, ```./NOVAa 500 1000 0.2 1``` should suffice)
