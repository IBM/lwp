# lwp

Lightweight profiling library

## Build

Just `make` should suffice.

## Run

### Run UMT on WSC

- Go to `codes/umt`;
- Try `./run.sh --help` for information;

Examples:

- Run without profiling, 1 node, 2 zones (small):
```
./run.sh --dir out --nodes 1 --zones 2
```

- Same setup, now with nvprof (summarized output):
```
./run.sh --dir out --nodes 1 --zones 2 --profiling nvp
```

- Run with nvprof (traces):
```
./run.sh --dir out --nodes 1 --zones 2 --profiling nvt
```

- Run with lwp:
```
./run.sh --dir out --nodes 1 --zones 2 --profiling lwp
```

Typically valid setups include zones between 2 and 32.
The run script will automatically generate the required input files
depending on the parameters selected.

## Troubleshooting

### XQuartz on MacOS

Running GUI programs over SSH typically requires a functioning installation of X server.
On Mac this is provided by XQuartz. 
`lwp_gui` uses SDL for graphics, which is based on OpenGL. 
OpenGL over SSH with newer XQuartz versions does not work:

```
X Error of failed request:  BadValue (integer parameter out of range for operation)
  Major opcode of failed request:  149 (GLX)
  Minor opcode of failed request:  3 (X_GLXCreateContext)
  Value in failed request:  0x0
  Serial number of failed request:  86
  Current serial number in output stream:  87
```

If you experience this issue, please downgrade to XQuartz 2.7.8.
The software package can be obtained [here](https://www.xquartz.org/releases/XQuartz-2.7.8.html).
Installation should be straightforward.
You can verify the currently installed version by running `xclock` from a Terminal on Mac.
Then, from the top menu bar select XQuartz > About XQuartz.
