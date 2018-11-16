"""Run all result scripts."""

import results.scripts.bland_table as bland_table
import results.scripts.combine_trials as combine_trials
import results.scripts.length_table as length_table
import results.scripts.plot_results as plot_results


combine_trials.main()

# Figures
plot_results.main()

# Tables
bland_table.main()
length_table.main()
