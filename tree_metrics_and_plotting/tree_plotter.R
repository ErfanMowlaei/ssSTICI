# ============================
# Plot tree tips by CNAclone
# with user-controlled colors,
# clone visibility toggles,
# rectangular/circular layout,
# dot vs pendant-edge coloring,
# and optional dropping of tips missing from the clone CSV
# ============================

# Install these once if needed:
# install.packages("ape")
# install.packages("ggplot2")
# install.packages("dplyr")
# install.packages("RColorBrewer")
#
# ggtree is usually installed through Bioconductor:
# if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
# BiocManager::install("ggtree")

library(ape)
library(ggtree)
library(ggplot2)
library(dplyr)
library(RColorBrewer)

# ----------------------------
# Input files
# ----------------------------

tree_file <- "~/tree.nwk"
clone_file <- "~/Tall5G7snv_CloneIDs.csv"

# ----------------------------
# Plotting variables
# ----------------------------

edge_width <- 0.1
branch_color <- "grey"

# Used when tip_label_mode <- "dots"
tip_dot_size <- 0.05

# Used when tip_label_mode <- "pendant_edges"
pendant_edge_width <- 1.0

plot_width <- 1
plot_height <- 2
plot_units <- "in"
plot_resolution <- 500

# Options: "png", "pdf", "tiff", "jpeg"
output_file_type <- "png"

# ----------------------------
# Tree layout variables
# ----------------------------
# Options:
# "rectangular" for a conventional rectangular tree
# "circular" for a circular tree

tree_layout <- "rectangular"

if (!tree_layout %in% c("rectangular", "circular")) {
  stop('tree_layout must be either "rectangular" or "circular".')
}

# ----------------------------
# Tip/edge clone display mode
# ----------------------------
# Options:
# "dots"           = color dots at the tips of pendant edges
# "pendant_edges"  = color the terminal pendant edge for each tip

tip_label_mode <- "dots"

if (!tip_label_mode %in% c("dots", "pendant_edges")) {
  stop('tip_label_mode must be either "dots" or "pendant_edges".')
}

# Color for CNAclone categories that are turned off
uncolored_tip_color <- "grey"

# Color for tips missing from the clone CSV
# This is only used when drop_tips_missing_from_csv <- FALSE.
missing_clone_color <- "grey"

# Should the CNAclone legend be shown in the final plot?
# TRUE  = show legend
# FALSE = hide legend
show_plot_legend <- FALSE

# Should turned-off categories appear in the legend?
# Only relevant when show_plot_legend <- TRUE.
show_uncolored_in_legend <- FALSE

# Should tree tips missing from the clone CSV be removed before plotting?
# TRUE  = remove unmatched tips using ape::drop.tip()
# FALSE = keep unmatched tips and color them with missing_clone_color
#
# This removes tree tips that are absent from the clone CSV.
# It does not remove clone CSV rows whose labels are absent from the tree.
drop_tips_missing_from_csv <- TRUE

# ----------------------------
# User-controlled CNAclone colors
# ----------------------------
# CNAclone_value must match values in the CNAclone column.
# Tip_color can be any R color name or hex code.
# Color_tips = TRUE means tips or pendant edges are colored using Tip_color.
# Color_tips = FALSE means they are plotted as uncolored_tip_color.

clone_display_settings <- data.frame(
  CNAclone_value = c(
    "2",
    "3",
    "4",
    "10",
    "11"
  ),
  Tip_color = c(
    "#92D050",
    "#E97132",
    "#196B24",
    "#0F9ED5",
    "#A02B93"
  ),
  Color_tips = c(
    FALSE,
    FALSE,
    FALSE,
    FALSE,
    TRUE
  ),
  stringsAsFactors = FALSE
)

# ----------------------------
# Read files
# ----------------------------

tree <- read.tree(tree_file)

# Always order nodes cladewise before plotting
tree <- reorder(tree, order = "cladewise")

clone_df <- read.csv(clone_file, stringsAsFactors = FALSE, check.names = FALSE)

# Confirm required columns exist
required_cols <- c("Cell", "CNAclone")
missing_cols <- setdiff(required_cols, colnames(clone_df))

if (length(missing_cols) > 0) {
  stop(
    paste(
      "The clone file is missing required column(s):",
      paste(missing_cols, collapse = ", ")
    )
  )
}

clone_df <- clone_df %>%
  mutate(
    Cell = as.character(Cell),
    CNAclone = as.character(CNAclone)
  )

# ----------------------------
# Check matching between tree tips and CSV labels
# and optionally drop unmatched tree tips
# ----------------------------

tree_tips <- tree$tip.label
csv_tips <- clone_df$Cell

tips_missing_from_csv <- setdiff(tree_tips, csv_tips)
csv_labels_not_in_tree <- setdiff(csv_tips, tree_tips)

message("Number of tree tips before optional filtering: ", length(tree_tips))
message("Number of rows in clone CSV: ", nrow(clone_df))
message("Tree tips missing from clone CSV: ", length(tips_missing_from_csv))
message("CSV labels not found in tree: ", length(csv_labels_not_in_tree))

if (drop_tips_missing_from_csv && length(tips_missing_from_csv) > 0) {
  tree <- drop.tip(tree, tips_missing_from_csv)

  # Reorder again after pruning
  tree <- reorder(tree, order = "cladewise")

  message("Dropped tree tips missing from clone CSV: ", length(tips_missing_from_csv))
  message("Number of tree tips after filtering: ", length(tree$tip.label))
}

# ----------------------------
# Validate clone display settings
# ----------------------------

observed_clones <- sort(unique(clone_df$CNAclone))
specified_clones <- clone_display_settings$CNAclone_value

clones_missing_from_settings <- setdiff(observed_clones, specified_clones)
settings_not_in_data <- setdiff(specified_clones, observed_clones)

if (length(clones_missing_from_settings) > 0) {
  warning(
    paste(
      "These CNAclone values are present in the CSV but missing from clone_display_settings:",
      paste(clones_missing_from_settings, collapse = ", "),
      "\nThey will be plotted as uncolored."
    )
  )
}

if (length(settings_not_in_data) > 0) {
  message(
    "These CNAclone values are listed in clone_display_settings but were not found in the CSV: ",
    paste(settings_not_in_data, collapse = ", ")
  )
}

# ----------------------------
# Apply clone color/toggle settings
# ----------------------------

clone_df <- clone_df %>%
  left_join(
    clone_display_settings,
    by = c("CNAclone" = "CNAclone_value")
  ) %>%
  mutate(
    Color_tips = ifelse(is.na(Color_tips), FALSE, Color_tips),
    CNAclone_plot = case_when(
      Color_tips ~ CNAclone,
      !Color_tips ~ "Not colored"
    )
  )

# ----------------------------
# Make output filename
# ----------------------------

tree_dir <- dirname(tree_file)
tree_base <- tools::file_path_sans_ext(basename(tree_file))

missing_tip_mode <- ifelse(
  drop_tips_missing_from_csv,
  "missingTipsDropped",
  "missingTipsKept"
)

legend_mode <- ifelse(
  show_plot_legend,
  "legendOn",
  "legendOff"
)

output_file <- file.path(
  tree_dir,
  paste0(
    tree_base,
    "_CNAclone_",
    tree_layout,
    "_",
    tip_label_mode,
    "_",
    missing_tip_mode,
    "_",
    legend_mode,
    ".",
    output_file_type
  )
)

# ----------------------------
# Build color scale
# ----------------------------

active_clone_settings <- clone_display_settings %>%
  filter(Color_tips) %>%
  select(CNAclone_value, Tip_color)

clone_colors <- active_clone_settings$Tip_color
names(clone_colors) <- active_clone_settings$CNAclone_value

plot_colors <- c(
  clone_colors,
  "Not colored" = uncolored_tip_color
)

legend_breaks <- names(plot_colors)

if (!show_uncolored_in_legend) {
  legend_breaks <- setdiff(legend_breaks, "Not colored")
}

# ----------------------------
# Plot tree
# ----------------------------

if (tip_label_mode == "dots") {

  # In dots mode, draw tip dots only for clone categories whose
  # Color_tips setting is TRUE. This prevents toggled-off clone
  # categories from receiving fallback "Not colored" dots.
  p <- ggtree(
    tree,
    layout = tree_layout,
    size = edge_width,
    color = branch_color
  ) %<+% clone_df +
    geom_tippoint(
      data = function(x) dplyr::filter(x, Color_tips %in% TRUE),
      aes(color = CNAclone_plot),
      size = tip_dot_size,
      na.rm = TRUE
    ) +
    scale_color_manual(
      values = clone_colors,
      breaks = legend_breaks,
      na.value = missing_clone_color,
      name = "CNAclone"
    )

} else if (tip_label_mode == "pendant_edges") {

  # Convert tree to the data structure used by ggtree.
  # For each edge, the child node is stored in the variable "node".
  # Tip nodes have labels matching tree$tip.label.
  p_base <- ggtree(
    tree,
    layout = tree_layout,
    size = edge_width,
    color = branch_color
  )

  tree_plot_data <- p_base$data

  # Add CNAclone information to terminal branches only.
  # Internal branches remain uncolored and use edge_width.
  tree_plot_data <- tree_plot_data %>%
    left_join(
      clone_df,
      by = c("label" = "Cell")
    ) %>%
    mutate(
      edge_clone_plot = case_when(
        isTip & !is.na(CNAclone_plot) ~ CNAclone_plot,
        isTip & is.na(CNAclone_plot) ~ "Missing clone assignment",
        TRUE ~ NA_character_
      ),
      edge_plot_width = case_when(
        isTip ~ pendant_edge_width,
        TRUE ~ edge_width
      )
    )

  p <- p_base
  p$data <- tree_plot_data

  p <- p +
    aes(color = edge_clone_plot, size = edge_plot_width) +
    scale_color_manual(
      values = c(
        plot_colors,
        "Missing clone assignment" = missing_clone_color
      ),
      breaks = legend_breaks,
      na.value = branch_color,
      name = "CNAclone"
    ) +
    scale_size_identity()

}

# ----------------------------
# Shared plot styling
# ----------------------------
# theme_tree() intentionally removes the branch-length axis.
# Use theme_tree2() instead only if you want the branch-length axis shown.

legend_position <- ifelse(show_plot_legend, "right", "none")

p <- p +
  theme_tree() +
  theme(
    legend.position = legend_position,
    legend.title = element_text(size = 11),
    legend.text = element_text(size = 10)
  )

# ----------------------------
# Save plot
# ----------------------------

ggsave(
  filename = output_file,
  plot = p,
  width = plot_width,
  height = plot_height,
  units = plot_units,
  dpi = plot_resolution,
  device = output_file_type
)

message("Saved plot to: ", output_file)
