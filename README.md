# MasterArbeit

# How to run the program

Run in terminal following command:

```python
PYTHON .\run_programm.py -c {config_file} -n {object_name} 
```

## Arguments
- Configuration File: There are several json files in the folder configuration, which contains all the important parameters for the generation of a 3D object (mesh, point cloud).

```python
-c .\config_files\config.json
```
- Object Name: the name of the object , that should be scan.

```python
-n apple
```

## Example 

```python
PYTHON .\run_programm.py -c .\config_files\config.json -n banana
```
