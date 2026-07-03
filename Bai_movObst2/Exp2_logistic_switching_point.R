# ======================================================================
#  Exp 2  (Cohen_movObst2) -- passing-order GLMM
#  Deliverables for Bill:
#    (1) logistic-regression output
#    (2) a logistic-function graph for EACH predictor
#    (3) the "switching point": predictor value where P(ahead) = 0.50
#        (here the focal predictor is PATH ANGLE)
# ======================================================================
library(dplyr)
library(lme4)
library(car)        # Anova
library(ggplot2)

set.seed(88)
setwd("C:/Users/baiji/OneDrive/Academic/Dissertation/Codes/Bai_movObst2")

data <- read.csv("Cohen_movObst2_pass_order.csv")

# Outcome: behind = 0, ahead = 1
data$pass_order[data$pass_order == -1] <- 0
# Use the magnitude of the path angle
data$obst_angle <- abs(data$obst_angle)

# ----------------------------------------------------------------------
# SECTION 1 -- categorical model (what you already ran) -> ANOVA
# ----------------------------------------------------------------------
data_cat <- data
data_cat$obst_angle <- factor(data_cat$obst_angle)
data_cat$obst_dist  <- factor(data_cat$obst_dist)

m_cat <- glmer(
  pass_order ~ obst_angle * obst_dist + (1 | subj_id),
  data = data_cat, family = binomial(link = "logit")
)
Anova(m_cat)        # Type-II Wald chi-square tests

# ----------------------------------------------------------------------
# SECTION 2 -- continuous model -> logistic curve + switching point
# Predictors stay numeric so a 50% crossing / smooth curve exist.
# ----------------------------------------------------------------------
data$obst_angle <- as.numeric(data$obst_angle)
data$obst_dist  <- as.numeric(data$obst_dist)

m_cont <- glmer(
  pass_order ~ obst_angle * obst_dist + (1 | subj_id),
  data = data, family = binomial(link = "logit")
)

# (1) Output ----------------------------------------------------------
summary(m_cont)     # fixed-effect coefficients (log-odds) = the logistic fit
Anova(m_cont)

# ----------------------------------------------------------------------
# SECTION 2b -- rescaled refit (clears the "rescale variables" warning)
# z-score the predictors so the fixed effects are well-conditioned.
# Predictions / switching points are unchanged; this just gives a clean
# coefficient table and standard errors for the manuscript.
# ----------------------------------------------------------------------
data$angle_z <- as.numeric(scale(data$obst_angle))
data$dist_z  <- as.numeric(scale(data$obst_dist))

m_z <- glmer(
  pass_order ~ angle_z * dist_z + (1 | subj_id),
  data = data, family = binomial(link = "logit")
)
summary(m_z)   # clean coefficients (per 1 SD of each predictor)
Anova(m_z)

# Switching angle in ORIGINAL units, confirmed from the rescaled model:
bz <- fixef(m_z)
a_mu <- mean(data$obst_angle); a_sd <- sd(data$obst_angle)
d_mu <- mean(data$obst_dist);  d_sd <- sd(data$obst_dist)
sw_angle_z <- function(dist) {
  dz <- (dist - d_mu) / d_sd
  az <- -(bz[["(Intercept)"]] + bz[["dist_z"]] * dz) /
         (bz[["angle_z"]] + bz[["angle_z:dist_z"]] * dz)
  az * a_sd + a_mu
}
dists0 <- sort(unique(data$obst_dist))
cat("\nSwitching angle (rescaled model, original units):\n")
print(data.frame(obst_dist = dists0, switch_angle = sapply(dists0, sw_angle_z)))

# (3) Switching point: PATH ANGLE at P(ahead) = 0.50 ------------------
# logit(p) = b0 + b1*angle + b2*dist + b3*angle*dist
# set = 0  ->  angle* = -(b0 + b2*dist) / (b1 + b3*dist)
b <- fixef(m_cont)
sw_angle <- function(dist)
  -(b[["(Intercept)"]] + b[["obst_dist"]] * dist) /
   (b[["obst_angle"]]  + b[["obst_angle:obst_dist"]] * dist)

dists <- sort(unique(data$obst_dist))
switch_points <- data.frame(obst_dist    = dists,
                            switch_angle = sapply(dists, sw_angle))
cat("\nSwitching path angle (P=0.50) at each initial distance:\n")
print(switch_points)

# ... and the reverse: INITIAL DISTANCE at P(ahead) = 0.50, per angle
# set logit = 0  ->  dist* = -(b0 + b1*angle) / (b2 + b3*angle)
sw_dist <- function(angle)
  -(b[["(Intercept)"]] + b[["obst_angle"]] * angle) /
   (b[["obst_dist"]]   + b[["obst_angle:obst_dist"]] * angle)

angles_all <- sort(unique(data$obst_angle))
switch_points_dist <- data.frame(obst_angle  = angles_all,
                                 switch_dist = sapply(angles_all, sw_dist))
cat("\nSwitching initial distance (P=0.50) at each path angle:\n")
print(switch_points_dist)

# Optional: one overall switching angle (ignore distance) -------------
# m_one <- glmer(pass_order ~ obst_angle + (1|subj_id), data, binomial)
# -fixef(m_one)[1] / fixef(m_one)[2]

# (2) Logistic-function graphs ----------------------------------------
predict_fixed <- function(model, grid, form) {
  X   <- model.matrix(form, grid)
  eta <- as.vector(X %*% fixef(model))
  se  <- sqrt(diag(X %*% as.matrix(vcov(model)) %*% t(X)))
  grid$p     <- plogis(eta)
  grid$lower <- plogis(eta - 1.96 * se)
  grid$upper <- plogis(eta + 1.96 * se)
  grid
}

obs <- data %>%
  group_by(obst_angle, obst_dist) %>%
  summarise(p = mean(pass_order), n = n(), .groups = "drop")

## Graph A: x = path angle, one curve per initial distance (the focal one)
gridA <- expand.grid(
  obst_angle = seq(min(data$obst_angle), max(data$obst_angle), length.out = 200),
  obst_dist  = dists
)
gridA <- predict_fixed(m_cont, gridA, ~ obst_angle * obst_dist)

pA <- ggplot(gridA, aes(obst_angle, p, color = factor(obst_dist),
                        fill = factor(obst_dist))) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = obs, aes(obst_angle, p, color = factor(obst_dist)),
             inherit.aes = FALSE, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_vline(data = switch_points,
             aes(xintercept = switch_angle, color = factor(obst_dist)),
             linetype = "dotted", show.legend = FALSE) +
  labs(x = "Path angle", y = "P(pass ahead)",
       color = "Initial distance", fill = "Initial distance",
       title = "Exp 2: logistic function over path angle") +
  theme_classic()
print(pA)

## Graph B: x = initial distance, one curve per path angle
angles <- sort(unique(data$obst_angle))
gridB <- expand.grid(
  obst_dist  = seq(min(data$obst_dist), max(data$obst_dist), length.out = 200),
  obst_angle = angles
)
gridB <- predict_fixed(m_cont, gridB, ~ obst_angle * obst_dist)

# only draw switching lines that fall within the tested distance range
sw_dist_in <- subset(switch_points_dist,
                     switch_dist >= min(data$obst_dist) &
                     switch_dist <= max(data$obst_dist))

pB <- ggplot(gridB, aes(obst_dist, p, color = factor(obst_angle),
                        fill = factor(obst_angle))) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.15, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = obs, aes(obst_dist, p, color = factor(obst_angle)),
             inherit.aes = FALSE, size = 2) +
  geom_hline(yintercept = 0.5, linetype = "dashed") +
  geom_vline(data = sw_dist_in,
             aes(xintercept = switch_dist, color = factor(obst_angle)),
             linetype = "dotted", show.legend = FALSE) +
  coord_cartesian(xlim = range(data$obst_dist)) +
  labs(x = "Initial distance", y = "P(pass ahead)",
       color = "Path angle", fill = "Path angle",
       title = "Exp 2: logistic function over initial distance") +
  theme_classic()
print(pB)

# Save figures (optional)
ggsave("Exp2_logistic_angle.png", pA, width = 6, height = 4, dpi = 300)
ggsave("Exp2_logistic_dist.png",  pB, width = 6, height = 4, dpi = 300)
