# ============================================================
# Whole-tree and per-clone metrics for .nwk / .newick trees
#
# Whole-tree metrics:
#   1. Colless' phylogeny imbalance
#   2. Edge ratio =
#        mean internal edge length / mean pendant edge length
#
# Per-clone metrics:
#   1. Colless' imbalance after pruning tree to clone tips
#   2. Neighbor-ness index
#   3. Out/within distance ratio
#
# Notes:
#   - Tips absent from the clone assignment CSV are dropped/ignored.
#   - Polytomies are resolved with:
#       ape::multi2di(tree, random = TRUE, equiprob = TRUE)
#   - Colless' imbalance is computed after polytomy resolution.
# ============================================================


# ----------------------------
# User settings
# ----------------------------

tree_dir <- "~/trees"

clone_csv <- "~/Tall5G7snv_CloneIDs.csv"
#clone_csv <- "~/OV025-copyKATCloneID.csv"
#clone_csv <- "~/TNBC5-copyKATCloneID.csv"

whole_tree_output_csv <- file.path(tree_dir, "tree_metrics_summary_by_conf.csv")

clone_output_csv <- file.path(tree_dir, "tree_clone_metrics_summary_by_conf.csv")

tree_extensions <- c("\\.nwk$", "\\.newick$", "\\.tree$", "\\.tre$")

# Set this to any integer for reproducible random polytomy resolution.
# Set to NULL if you do not want to set a seed.
random_seed <- 123

# Column names in the clone assignment CSV
tip_column <- "Cell"
clone_column <- "CNAclone"


# ----------------------------
# Package setup
# ----------------------------

required_packages <- c("ape", "apTreeshape")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

library(ape)
library(apTreeshape)


# ----------------------------
# Reproducibility
# ----------------------------

if (!is.null(random_seed)) {
  set.seed(random_seed)
}


# ----------------------------
# Input clone table
# ----------------------------

clone_table <- read.csv(clone_csv, stringsAsFactors = FALSE)

if (!(tip_column %in% colnames(clone_table))) {
  stop("Could not find tip column in clone CSV: ", tip_column)
}

if (!(clone_column %in% colnames(clone_table))) {
  stop("Could not find clone column in clone CSV: ", clone_column)
}

clone_table <- clone_table[, c(tip_column, clone_column)]
colnames(clone_table) <- c("tip", "clone")

clone_table$tip <- as.character(clone_table$tip)
clone_table$clone <- as.character(clone_table$clone)

clone_table <- clone_table[!is.na(clone_table$tip), ]
clone_table <- clone_table[!is.na(clone_table$clone), ]
clone_table <- clone_table[clone_table$tip != "", ]

# If duplicate tip assignments exist, keep the first and warn.
duplicated_tips <- clone_table$tip[duplicated(clone_table$tip)]

if (length(duplicated_tips) > 0) {
  warning(
    "Duplicate tip IDs found in clone CSV. Keeping first assignment for each duplicated tip."
  )
  clone_table <- clone_table[!duplicated(clone_table$tip), ]
}

all_clone_values <- sort(unique(clone_table$clone))


# ----------------------------
# Helper functions
# ----------------------------

get_tree_files <- function(directory, extensions) {
  pattern <- paste(extensions, collapse = "|")
  
  list.files(
    path = directory,
    pattern = pattern,
    full.names = TRUE,
    ignore.case = TRUE
  )
}


get_root_degree <- function(tree) {
  n_tips <- length(tree$tip.label)
  root_node <- n_tips + 1
  
  sum(tree$edge[, 1] == root_node)
}


has_polytomies <- function(tree) {
  parent_nodes <- tree$edge[, 1]
  child_counts <- table(parent_nodes)
  
  any(child_counts > 2)
}


resolve_polytomies_if_needed <- function(tree) {
  if (has_polytomies(tree)) {
    tree <- ape::multi2di(
      phy = tree,
      random = TRUE,
      equiprob = TRUE
    )
  }
  
  tree
}


drop_unassigned_tips <- function(tree, clone_table) {
  assigned_tips <- clone_table$tip
  tips_to_drop <- setdiff(tree$tip.label, assigned_tips)
  
  if (length(tips_to_drop) > 0) {
    tree <- ape::drop.tip(tree, tips_to_drop)
  }
  
  tree
}


compute_colless <- function(tree) {
  tree_shape <- apTreeshape::as.treeshape(tree)
  
  apTreeshape::colless(tree_shape)
}


compute_edge_ratio <- function(tree) {
  if (is.null(tree$edge.length)) {
    return(NA_real_)
  }
  
  edge_lengths <- tree$edge.length
  n_tips <- length(tree$tip.label)
  
  child_nodes <- tree$edge[, 2]
  
  # In ape phylo objects:
  # tips are numbered 1:n_tips
  # internal nodes are numbered n_tips + 1, n_tips + 2, etc.
  pendant_edges <- child_nodes <= n_tips
  internal_edges <- child_nodes > n_tips
  
  pendant_lengths <- edge_lengths[pendant_edges]
  internal_lengths <- edge_lengths[internal_edges]
  
  pendant_lengths <- pendant_lengths[!is.na(pendant_lengths)]
  internal_lengths <- internal_lengths[!is.na(internal_lengths)]
  
  if (length(pendant_lengths) == 0 || length(internal_lengths) == 0) {
    return(NA_real_)
  }
  
  mean_pendant <- mean(pendant_lengths)
  mean_internal <- mean(internal_lengths)
  
  if (mean_pendant == 0) {
    return(NA_real_)
  }
  
  mean_internal / mean_pendant
}


get_descendant_tips <- function(tree, node) {
  n_tips <- length(tree$tip.label)
  
  if (node <= n_tips) {
    return(tree$tip.label[node])
  }
  
  children <- tree$edge[tree$edge[, 1] == node, 2]
  
  descendant_tips <- unlist(
    lapply(children, function(child) {
      get_descendant_tips(tree, child)
    }),
    use.names = FALSE
  )
  
  descendant_tips
}


get_sister_tip_set_for_tip <- function(tree, tip_label) {
  tip_index <- match(tip_label, tree$tip.label)
  
  if (is.na(tip_index)) {
    return(character(0))
  }
  
  parent_node <- tree$edge[tree$edge[, 2] == tip_index, 1]
  
  if (length(parent_node) == 0) {
    return(character(0))
  }
  
  sibling_nodes <- tree$edge[tree$edge[, 1] == parent_node, 2]
  sibling_nodes <- sibling_nodes[sibling_nodes != tip_index]
  
  sister_tips <- unlist(
    lapply(sibling_nodes, function(node) {
      get_descendant_tips(tree, node)
    }),
    use.names = FALSE
  )
  
  sister_tips
}


compute_neighborness_index <- function(tree, clone_assignments, target_clone) {
  target_tips <- names(clone_assignments)[clone_assignments == target_clone]
  target_tips <- intersect(target_tips, tree$tip.label)
  
  if (length(target_tips) == 0) {
    return(NA_real_)
  }
  
  hits <- rep(NA, length(target_tips))
  
  for (i in seq_along(target_tips)) {
    tip <- target_tips[i]
    
    sister_tips <- get_sister_tip_set_for_tip(tree, tip)
    
    # If no sister set exists, score as NA.
    if (length(sister_tips) == 0) {
      hits[i] <- NA
      next
    }
    
    sister_clones <- clone_assignments[sister_tips]
    sister_clones <- sister_clones[!is.na(sister_clones)]
    
    if (length(sister_clones) == 0) {
      hits[i] <- NA
      next
    }
    
    proportion_same_clone <- mean(sister_clones == target_clone)
    
    hits[i] <- proportion_same_clone >= 0.5
  }
  
  if (all(is.na(hits))) {
    return(NA_real_)
  }
  
  mean(hits, na.rm = TRUE)
}


compute_out_within_distance_ratio <- function(tree, clone_assignments, target_clone) {
  if (is.null(tree$edge.length)) {
    return(NA_real_)
  }
  
  target_tips <- names(clone_assignments)[clone_assignments == target_clone]
  target_tips <- intersect(target_tips, tree$tip.label)
  
  other_tips <- setdiff(tree$tip.label, target_tips)
  
  # Need at least two target tips to compute within-clone distances.
  if (length(target_tips) < 2) {
    return(NA_real_)
  }
  
  # Need at least one non-target tip to compute out-clone distances.
  if (length(other_tips) < 1) {
    return(NA_real_)
  }
  
  distance_matrix <- ape::cophenetic.phylo(tree)
  
  within_matrix <- distance_matrix[target_tips, target_tips, drop = FALSE]
  
  within_distances <- within_matrix[upper.tri(within_matrix)]
  
  out_matrix <- distance_matrix[target_tips, other_tips, drop = FALSE]
  
  out_distances <- as.vector(out_matrix)
  
  within_distances <- within_distances[!is.na(within_distances)]
  out_distances <- out_distances[!is.na(out_distances)]
  
  if (length(within_distances) == 0 || length(out_distances) == 0) {
    return(NA_real_)
  }
  
  mean_within <- mean(within_distances)
  mean_out <- mean(out_distances)
  
  if (mean_within == 0) {
    return(NA_real_)
  }
  
  mean_out / mean_within
}


make_clone_assignment_vector <- function(clone_table, tree) {
  clone_table_tree <- clone_table[clone_table$tip %in% tree$tip.label, ]
  
  clone_assignments <- clone_table_tree$clone
  names(clone_assignments) <- clone_table_tree$tip
  
  clone_assignments
}


read_first_tree_from_file <- function(file_path) {
  tree <- ape::read.tree(file_path)
  
  if (inherits(tree, "multiPhylo")) {
    tree <- tree[[1]]
  }
  
  tree
}


compute_whole_tree_metrics_for_file <- function(file_path, clone_table) {
  result <- data.frame(
    file_name = basename(file_path),
    file_path = file_path,
    
    n_tips_original = NA_integer_,
    n_tips_after_dropping_unassigned = NA_integer_,
    n_tips_dropped_unassigned = NA_integer_,
    
    root_degree_original = NA_integer_,
    is_binary_original = NA,
    had_polytomies_original = NA,
    
    root_degree_used = NA_integer_,
    is_binary_used = NA,
    had_polytomies_after_dropping_unassigned = NA,
    polytomies_resolved_with_multi2di = NA,
    
    colless_imbalance = NA_real_,
    edge_ratio = NA_real_,
    
    status = "success",
    stringsAsFactors = FALSE
  )
  
  tree_original <- tryCatch(
    read_first_tree_from_file(file_path),
    error = function(e) {
      result$status <- paste("read.tree error:", conditionMessage(e))
      return(NULL)
    }
  )
  
  if (is.null(tree_original)) {
    return(result)
  }
  
  result$n_tips_original <- length(tree_original$tip.label)
  result$root_degree_original <- get_root_degree(tree_original)
  result$is_binary_original <- ape::is.binary.phylo(tree_original)
  result$had_polytomies_original <- has_polytomies(tree_original)
  
  tree_assigned <- tryCatch(
    drop_unassigned_tips(tree_original, clone_table),
    error = function(e) {
      result$status <<- paste("drop unassigned tips error:", conditionMessage(e))
      return(NULL)
    }
  )
  
  if (is.null(tree_assigned)) {
    return(result)
  }
  
  result$n_tips_after_dropping_unassigned <- length(tree_assigned$tip.label)
  result$n_tips_dropped_unassigned <-
    result$n_tips_original - result$n_tips_after_dropping_unassigned
  
  if (length(tree_assigned$tip.label) < 2) {
    result$status <- "error: fewer than 2 assigned tips remain after dropping unassigned tips"
    return(result)
  }
  
  result$had_polytomies_after_dropping_unassigned <- has_polytomies(tree_assigned)
  
  tree_used <- tryCatch(
    resolve_polytomies_if_needed(tree_assigned),
    error = function(e) {
      result$status <<- paste("multi2di error:", conditionMessage(e))
      return(NULL)
    }
  )
  
  if (is.null(tree_used)) {
    return(result)
  }
  
  result$polytomies_resolved_with_multi2di <-
    result$had_polytomies_after_dropping_unassigned
  
  result$root_degree_used <- get_root_degree(tree_used)
  result$is_binary_used <- ape::is.binary.phylo(tree_used)
  
  result$edge_ratio <- tryCatch(
    compute_edge_ratio(tree_used),
    error = function(e) {
      result$status <<- paste("edge_ratio error:", conditionMessage(e))
      NA_real_
    }
  )
  
  result$colless_imbalance <- tryCatch({
    if (!ape::is.binary.phylo(tree_used)) {
      stop(
        "Colless skipped: tree is still not binary after multi2di; root degree = ",
        result$root_degree_used
      )
    }
    
    compute_colless(tree_used)
  }, error = function(e) {
    if (result$status == "success") {
      result$status <<- paste("colless error:", conditionMessage(e))
    } else {
      result$status <<- paste(result$status, "| colless error:", conditionMessage(e))
    }
    
    NA_real_
  })
  
  result
}


compute_clone_metrics_for_file <- function(file_path, clone_table, all_clone_values) {
  tree_original <- tryCatch(
    read_first_tree_from_file(file_path),
    error = function(e) {
      return(NULL)
    }
  )
  
  if (is.null(tree_original)) {
    return(data.frame(
      file_name = basename(file_path),
      file_path = file_path,
      clone = all_clone_values,
      n_clone_tips = NA_integer_,
      n_tree_tips_after_dropping_unassigned = NA_integer_,
      clone_colless_imbalance = NA_real_,
      neighborness_index = NA_real_,
      out_within_distance_ratio = NA_real_,
      clone_status = paste("read.tree error"),
      stringsAsFactors = FALSE
    ))
  }
  
  tree_assigned <- tryCatch(
    drop_unassigned_tips(tree_original, clone_table),
    error = function(e) {
      return(NULL)
    }
  )
  
  if (is.null(tree_assigned) || length(tree_assigned$tip.label) < 2) {
    return(data.frame(
      file_name = basename(file_path),
      file_path = file_path,
      clone = all_clone_values,
      n_clone_tips = NA_integer_,
      n_tree_tips_after_dropping_unassigned = NA_integer_,
      clone_colless_imbalance = NA_real_,
      neighborness_index = NA_real_,
      out_within_distance_ratio = NA_real_,
      clone_status = "error: fewer than 2 assigned tips remain after dropping unassigned tips",
      stringsAsFactors = FALSE
    ))
  }
  
  # Resolve polytomies in the full assigned tree before topology-based metrics.
  tree_full_used <- resolve_polytomies_if_needed(tree_assigned)
  
  clone_assignments <- make_clone_assignment_vector(clone_table, tree_full_used)
  
  rows <- list()
  
  for (target_clone in all_clone_values) {
    target_tips <- names(clone_assignments)[clone_assignments == target_clone]
    target_tips <- intersect(target_tips, tree_full_used$tip.label)
    
    row <- data.frame(
      file_name = basename(file_path),
      file_path = file_path,
      clone = target_clone,
      
      n_clone_tips = length(target_tips),
      n_tree_tips_after_dropping_unassigned = length(tree_full_used$tip.label),
      
      clone_had_polytomies_after_pruning = NA,
      clone_polytomies_resolved_with_multi2di = NA,
      clone_is_binary_used = NA,
      
      clone_colless_imbalance = NA_real_,
      neighborness_index = NA_real_,
      out_within_distance_ratio = NA_real_,
      
      clone_status = "success",
      stringsAsFactors = FALSE
    )
    
    # ----------------------------
    # Neighbor-ness index
    # Computed on full assigned tree.
    # ----------------------------
    
    row$neighborness_index <- tryCatch(
      compute_neighborness_index(
        tree = tree_full_used,
        clone_assignments = clone_assignments,
        target_clone = target_clone
      ),
      error = function(e) {
        row$clone_status <<- paste("neighborness error:", conditionMessage(e))
        NA_real_
      }
    )
    
    # ----------------------------
    # Out/within distance ratio
    # Computed on full assigned tree.
    # ----------------------------
    
    row$out_within_distance_ratio <- tryCatch(
      compute_out_within_distance_ratio(
        tree = tree_full_used,
        clone_assignments = clone_assignments,
        target_clone = target_clone
      ),
      error = function(e) {
        if (row$clone_status == "success") {
          row$clone_status <<- paste("out_within_distance_ratio error:", conditionMessage(e))
        } else {
          row$clone_status <<- paste(
            row$clone_status,
            "| out_within_distance_ratio error:",
            conditionMessage(e)
          )
        }
        
        NA_real_
      }
    )
    
    # ----------------------------
    # Clone-level Colless imbalance
    # Computed after pruning to target clone tips.
    # ----------------------------
    
    row$clone_colless_imbalance <- tryCatch({
      if (length(target_tips) < 2) {
        stop("Colless skipped: fewer than 2 tips in clone")
      }
      
      tips_to_drop <- setdiff(tree_full_used$tip.label, target_tips)
      clone_tree <- ape::drop.tip(tree_full_used, tips_to_drop)
      
      row$clone_had_polytomies_after_pruning <<- has_polytomies(clone_tree)
      
      clone_tree_used <- resolve_polytomies_if_needed(clone_tree)
      
      row$clone_polytomies_resolved_with_multi2di <<-
        row$clone_had_polytomies_after_pruning
      
      row$clone_is_binary_used <<- ape::is.binary.phylo(clone_tree_used)
      
      if (!ape::is.binary.phylo(clone_tree_used)) {
        stop("Colless skipped: clone tree is still not binary after multi2di")
      }
      
      compute_colless(clone_tree_used)
    }, error = function(e) {
      if (row$clone_status == "success") {
        row$clone_status <<- paste("clone Colless error:", conditionMessage(e))
      } else {
        row$clone_status <<- paste(
          row$clone_status,
          "| clone Colless error:",
          conditionMessage(e)
        )
      }
      
      NA_real_
    })
    
    rows[[length(rows) + 1]] <- row
  }
  
  do.call(rbind, rows)
}


# ----------------------------
# Main script
# ----------------------------

tree_files <- get_tree_files(tree_dir, tree_extensions)

if (length(tree_files) == 0) {
  stop("No tree files found in the specified directory.")
}

whole_tree_metrics_list <- lapply(
  tree_files,
  compute_whole_tree_metrics_for_file,
  clone_table = clone_table
)

whole_tree_metrics_table <- do.call(rbind, whole_tree_metrics_list)

clone_metrics_list <- lapply(
  tree_files,
  compute_clone_metrics_for_file,
  clone_table = clone_table,
  all_clone_values = all_clone_values
)

clone_metrics_table <- do.call(rbind, clone_metrics_list)

write.csv(whole_tree_metrics_table, whole_tree_output_csv, row.names = FALSE)

write.csv(clone_metrics_table, clone_output_csv, row.names = FALSE)

cat("Done.\n")
cat("Processed", length(tree_files), "tree file(s).\n")
cat("Whole-tree results written to:\n", whole_tree_output_csv, "\n")
cat("Per-clone results written to:\n", clone_output_csv, "\n")