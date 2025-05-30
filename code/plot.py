from utils.plotter import Plotter
plotter = Plotter()

# plotter.plot_from_files(
#     [
#     ("SGD_0.01_01-25_04-49", "SGD"), 
#     ("SVRG_0.01_01-25_04-42", "SVRG"),
#     ("ShuffleSVRG_0.01_02-01_15-01", "ShuffleSVRG"),
#     ]
# )

plotter.plot_latest(["SGD", "SVRG",  "ShuffleSVRG", "NFGSVRG"])
