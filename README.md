# Not yet finished, but close \\[*]/

To install, in project directory, do: 
```
pip install .
```

### File syntax
Inside the file, you can define all configurations for which you want performance to be measured.  
```
# This is a comment
gmx mdrun -s file.tpr -ntomp 6 -gpu_id 0
gmx mdrun -s file.tpr -ntomp 12 -gpu_id 1
...
gmx mdrun -s file.tpr -ntomp 18 -gpu_id 0,1
```
Comments with # and empty lines are allowed and will be ignored during parsing.

Multiple configurations can be declared in a single line using following syntax:
```markdown
gmx mdrun -s file.tpr -ntomp 6|12

# changes into...
gmx mdrun -s file.tpr -ntomp 6
gmx mdrun -s file.tpr -ntomp 12
```

Vertical bars (|) can be combined, changing into all possible combinations:
```markdown
gmx mdrun -s file.tpr -ntomp 6|12 -update cpu|gpu

# changes into...
gmx mdrun -s file.tpr -ntomp 6 -update cpu
gmx mdrun -s file.tpr -ntomp 12 -update cpu
gmx mdrun -s file.tpr -ntomp 6 -update gpu
gmx mdrun -s file.tpr -ntomp 12 -update gpu
```

To filter out unwanted combinations, bang symbol (!) can be used. All configurations containing arguments defined after it will be filtered out. Order of the arguments does not matter.

```markdown
gmx mdrun -s file.tpr -ntomp 6|12 -update cpu|gpu
! -ntomp 6 -update cpu

# changes into...
gmx mdrun -s file.tpr -ntomp 12 -update cpu
gmx mdrun -s file.tpr -ntomp 6 -update gpu
gmx mdrun -s file.tpr -ntomp 12 -update gpu
```

When setting up many configurations, it might be beneficial to apply a filter only to specific lines. This can be done with #tags.
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

> [!IMPORTANT]
> Looking for ideas how to define paralell simulations...