# SPDX-FileCopyrightText: 2026 Piotr Gródek and the RadVeil contributors
# SPDX-License-Identifier: Apache-2.0
#
# mri_deface (Bischoff-Grethe et al. 2007) as a baseline. FreeSurfer no
# longer serves the files; scripts/fetch_containers.sh fetches the 2025
# Internet Archive captures of surfer.nmr.mgh.harvard.edu/pub/dist/mri_deface/
# and checks their SHA-256 before building with that directory as context.
# The binary is x86-64 only: on arm64 hosts it runs under emulation.
FROM ubuntu@sha256:98510ac07bb4545e389be502a8918b385c018bfe4ceeb5ac7e0375c782268b04
COPY mri_deface_linux /usr/local/bin/mri_deface
COPY talairach_mixed_with_skull.gca face.gca /opt/mri_deface/
RUN chmod +x /usr/local/bin/mri_deface && apt-get update && apt-get install -y --no-install-recommends libgomp1=12.3.0-1ubuntu1~22.04.3 && rm -rf /var/lib/apt/lists/*
WORKDIR /data
ENTRYPOINT ["mri_deface"]
