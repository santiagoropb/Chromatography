This is Pychrom!
======================

[![Build Status](https://travis-ci.org/santiagoropb/Chromatography.svg?branch=master)](https://travis-ci.org/santiagoropb/Chromatography)

Python package for chromatography modeling and optimization.

The design features of this package revolve around the need to have different "modellers" for the same "model". A model is an abstraction of a chromatography process. A modeller is the implementation of that abstraction in a particular "engine" or "solver". At the time of writing there are two modellers: CADET and Pyomo. The former uses a MOL+IVP strategy to solve the PDAE system. The latter uses a NLA (full discretization) strategy. This project aims at providing an abstration that allows the definition of a model in an abstract way, and the implementation (or interpretation) of that model in different modellers, in order to enable the comparison and contrasting of alternative modeling approaches for simulation and optimization.

Organization
------------

Modeling for Preparative Chromatography



