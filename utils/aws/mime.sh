<<<<<<< HEAD
=======
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

>>>>>>> ultralytics/master
# AWS EC2 instance startup 'MIME' script https://aws.amazon.com/premiumsupport/knowledge-center/execute-user-data-ec2/
# This script will run on every instance restart, not only on first start
# --- DO NOT COPY ABOVE COMMENTS WHEN PASTING INTO USERDATA ---

<<<<<<< HEAD
Content-Type: multipart/mixed; boundary="//"
MIME-Version: 1.0

--//
Content-Type: text/cloud-config; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename="cloud-config.txt"
=======
Content-Type: multipart/mixed
boundary="//"
MIME-Version: 1.0

--//
Content-Type: text/cloud-config
charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment
filename="cloud-config.txt"
>>>>>>> ultralytics/master

#cloud-config
cloud_final_modules:
- [scripts-user, always]

--//
<<<<<<< HEAD
Content-Type: text/x-shellscript; charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment; filename="userdata.txt"
=======
Content-Type: text/x-shellscript
charset="us-ascii"
MIME-Version: 1.0
Content-Transfer-Encoding: 7bit
Content-Disposition: attachment
filename="userdata.txt"
>>>>>>> ultralytics/master

#!/bin/bash
# --- paste contents of userdata.sh here ---
--//
