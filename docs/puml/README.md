# Creating PNG files

The PNG files for the UML diagrams are automatically generated by the docs GitHub action using the source files in this folder. In case you need to do this manually, and assuming you have installed [PlantUML](https://plantuml.com/) in a `/plantuml` folder, you can generate the PNG files with the following command:

```bash
$ java -DPLANTUML_LIMIT_SIZE=8192 -jar /plantuml/plantuml.jar -o img/ .
```

Alternatively, if you have installed the PlantUML binary e.g. via [HomeBrew](https://brew.sh/), etc.:

```bash
$ plantuml -v -tpng docs/puml/*.puml -o img -DPLANTUML_LIMIT_SIZE=8192
```