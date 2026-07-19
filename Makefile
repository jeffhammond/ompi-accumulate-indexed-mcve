# Build/test the single-file pure-MPI reproducer.  No libarmci, no -I.
#   make            # build vacc_mcve
#   make run        # run it NRUNS times on 2 nodes x 2 ppn and classify
#   make clean
# Override on the command line, e.g.:
#   make run MPICC=mpicc NP=4 LAUNCH="mpirun -n 4 --map-by ppr:2:node -x ARMCI_VERBOSE -x UCX_TLS -x UCX_NET_DEVICES" NRUNS=20
MPICC  ?= mpicc
CFLAGS ?= -O2
LDLIBS ?= -lm
NP     ?= 4
NRUNS  ?= 10
TIMEOUT?= 60
LAUNCH ?= mpirun -n $(NP) --map-by ppr:2:node -x ARMCI_VERBOSE

vacc_mcve: vacc_mcve.c
	$(MPICC) $(CFLAGS) $< -o $@ $(LDLIBS)

run: vacc_mcve
	@ok=0; bad=0; for r in $$(seq 1 $(NRUNS)); do \
	  out=$$(ARMCI_VERBOSE=1 timeout $(TIMEOUT) $(LAUNCH) ./vacc_mcve 2>&1); \
	  if echo "$$out" | grep -qa 'DONE OK'; then ok=$$((ok+1)); echo "run $$r: OK"; \
	  elif echo "$$out" | grep -qaE 'ERROR:|Bailing out'; then bad=$$((bad+1)); \
	       echo "run $$r: CORRUPT  $$(echo "$$out" | grep -m1 -a 'ERROR:')"; \
	  else bad=$$((bad+1)); echo "run $$r: HANG/other"; fi; \
	done; echo "==> OK=$$ok  REPRODUCED(corrupt/hang)=$$bad  of $(NRUNS)"

clean:
	rm -f vacc_mcve *.gcno *.gcda *.gcov

.PHONY: run clean
