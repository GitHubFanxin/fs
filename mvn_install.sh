#!/bin/bash
mvn install:install-file -DgroupId=pers.xin -DartifactId=fs -Dversion=1.0 -Dpackaging=jar -Dfile=./target/fs-1.0.jar
mvn install:install-file -DgroupId=pers.xin -DartifactId=fs -Dversion=1.0 -Dpackaging=jar -Dfile=./target/fs-1.0-sources.jar -Dclassifier=sources
mvn install:install-file -DgroupId=pers.xin -DartifactId=fs -Dversion=1.0 -Dpackaging=jar -Dfile=./target/fs-1.0-javadoc.jar -Dclassifier=javadoc
