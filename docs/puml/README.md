# Creating PNG files

The PNG files in folder `img` must be generated manually from the PUML files. Assuming you have installed PlantUML in a `/plantuml` folder, you can generate the PNG files with the following command:

```bash
java -DPLANTUML_LIMIT_SIZE=8192 -jar /plantuml/plantuml.jar -o img/ .
```
