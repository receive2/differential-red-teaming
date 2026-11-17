# SPDX-License-Identifier: Apache-2.0

"""The scoring side of the testbed: run eval cases against a subject tree over the frozen
ESCI world, grade them with code graders (and a pinned judge for rubrics), and apply the
edit sets that turn one tree into another (the reference implementation, a knockout)."""
