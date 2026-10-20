TECTONIC ?= tectonic
PYTHON ?= python3
PDFPC ?= pdfpc

.PHONY: all pdf notes tutorial present rehearse
all: pdf notes
pdf:
	$(TECTONIC) --keep-logs research-update.tex
notes:
	$(PYTHON) make_notes.py
tutorial:
	$(TECTONIC) --keep-logs tutorial.tex
present:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=9 --note-format=plain --windowed=both research-update.pdf
rehearse:
	$(PDFPC) --pdfpc-location=research-update.pdfpc --duration=9 --note-format=plain --single-screen --windowed=none research-update.pdf
