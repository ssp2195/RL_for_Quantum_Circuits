.PHONY: all figures clean distclean

all: main.pdf

main.pdf: main.tex preamble.tex references.bib $(wildcard sections/*.tex) $(wildcard figures/*)
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures:
	python scripts/make_figures.py

clean:
	latexmk -c main.tex

distclean:
	latexmk -C main.tex
