# Introduction

To install, do: 
```
git clone https://github.com/BeepBoopRun/gmx_tracker
cd gmx_tracker
pip3 install .
```

To use, do for example: 
```
python3 -m gmx_tracker -e --sim-directory SOME_PATH/sim_files/ -c "gmx mdrun -s file.tpr -ntomp 8|16|24"
```
This command will run 3 GROMACS simulations:
```
gmx mdrun -s file.tpr -ntomp 8
gmx mdrun -s file.tpr -ntomp 16
gmx mdrun -s file.tpr -ntomp 24
```
For each performance will be measured and displayed in real-time.

## Why use this?
With new versions of GROMACS, there are more and more options we can choose from to get extra performance or efficiency. We belive that this tool can be useful to navigate those options. It allows one to tersly describe many different configurations you might want to try. 

In the example above, the user can quickly check how many threads are actually needed for performance. This is useful in queue systems like SLURM, where additional resources might come with actual costs or queing penalties.  

Similar tests can be done in systems with many graphics cards, for example in a system with MIG set up. By specifing __-gpu_id 0|1|2|3__ user can see how performant the simulation would be on all instances, later choosing one that bests suits their needs. 

## File syntax
#### Inside the file, you can define all configurations for which you want performance to be measured.  
```markdown
gmx mdrun -s file.tpr -ntomp 6 -gpu_id 0
# This is a comment
gmx mdrun -s file.tpr -ntomp 12 -gpu_id 1
gmx mdrun -s file.tpr -ntomp 18 -gpu_id 0,1
```
#### Comments with # and empty lines are allowed and will be ignored during parsing.

#### Multiple configurations can be declared in a single line using following syntax:
```markdown
gmx mdrun -s file.tpr -ntomp 6|12

# changes into...
gmx mdrun -s file.tpr -ntomp 6
gmx mdrun -s file.tpr -ntomp 12
```

#### Vertical bars (|) can be combined, changing into all possible combinations:
```markdown
gmx mdrun -s file.tpr -ntomp 6|12 -update cpu|gpu

# changes into...
gmx mdrun -s file.tpr -ntomp 6 -update cpu
gmx mdrun -s file.tpr -ntomp 12 -update cpu
gmx mdrun -s file.tpr -ntomp 6 -update gpu
gmx mdrun -s file.tpr -ntomp 12 -update gpu
```

#### To filter out unwanted combinations, bang symbol (!) can be used. All configurations containing arguments defined after it will be filtered out. Order of the arguments does not matter.

```markdown
gmx mdrun -s file.tpr -ntomp 6|12 -update cpu|gpu
! -ntomp 6 -update cpu

# changes into...
gmx mdrun -s file.tpr -ntomp 12 -update cpu
gmx mdrun -s file.tpr -ntomp 6 -update gpu
gmx mdrun -s file.tpr -ntomp 12 -update gpu
```

#### When setting up many configurations, it might be beneficial to apply a filter only to specific lines. This can be done with #tags.
```markdown
gmx mdrun -s file.tpr -ntomp 32 -bonded cpu -update cpu
gmx mdrun -s file.tpr -ntomp 6 -bonded cpu|gpu -update cpu|gpu #tag1
! -bonded cpu -update cpu #tag1

# changes into...
gmx mdrun -s file.tpr -ntomp 32 -bonded cpu -update cpu
gmx mdrun -s file.tpr -ntomp 6 -bonded gpu -update gpu
gmx mdrun -s file.tpr -ntomp 6 -bonded cpu -update gpu
gmx mdrun -s file.tpr -ntomp 6 -bonded gpu -update cpu
```


#### Configurations that are right next to each other and share the @ symbol will be run in paralell.
```markdown
@gmx mdrun -s file.tpr -ntomp 6 -bonded cpu -update cpu
@gmx mdrun -s file.tpr -ntomp 6 -bonded cpu -update cpu

@gmx mdrun -s file.tpr -ntomp 12 -bonded cpu -update cpu
@gmx mdrun -s file.tpr -ntomp 12 -bonded cpu -update cpu
```
#### First pair will run in paralell, as well as the second pair. Empty spaces separate what should run together.

#### Symbols can be combined. The following is correct syntax:
```markdown
@gmx mdrun -s file.tpr -ntomp 6|12 -bonded cpu -update cpu
@gmx mdrun -s file.tpr -ntomp 6|12 -bonded cpu -update cpu

# changes into...
@gmx mdrun -s file.tpr -ntomp 6 -bonded cpu -update cpu
@gmx mdrun -s file.tpr -ntomp 6 -bonded cpu -update cpu

@gmx mdrun -s file.tpr -ntomp 6 -bonded cpu -update cpu
@gmx mdrun -s file.tpr -ntomp 12 -bonded cpu -update cpu

@gmx mdrun -s file.tpr -ntomp 12 -bonded cpu -update cpu
@gmx mdrun -s file.tpr -ntomp 12 -bonded cpu -update cpu
```
#### Simulations are deduplicated by default. 


> [!IMPORTANT]
> Not sure how to specify enviromental variables in the file... 
