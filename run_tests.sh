#!/bin/sh

nosetests --with-cov --cov-report html --cov=viztricks/ viztricks/tests/ && coverage report

