.. _intro-tutorial:

==========================
Contributing to OpenMRSLab
==========================

#############################################
Step 1. Fork and clone repository from GitHub
#############################################
#. Go to the OpenMRSLab GitHub `repository <https://github.com/openmrslab/openmrslab>`_.
#. Click on the **Fork** button at the top right of the page.

#######################
Step 2. Clone your fork
#######################
#. Ensure `git <https://git-scm.com/>`_ is installed on your computer.
#. On the GitHub page of your fork, click the green **Clone** button.
#. Copy the displayed URL.
#. Open Terminal and *cd* into the repository where you would like to clone your fork.
#. Enter the command: ``git clone [URL]``, where **[URL]** is the URL you copied in the step 3.

######################################
Step 3. Push code changes to your fork
######################################
#. Make your code changes.
#. Push your code to your fork using standard git practices:
    #. Open Terminal and *cd* into your fork directory
    #. Add all of your code changes: ``git add -A``
    #. Commit your code with a commit message: ``git commit -m "[COMMIT MESSAGE]"``
    #. Push your code: ``git push``

###################################################################################
Step 4. Create pull request to from your fork to the original OpenMRSLab repository
###################################################################################
#. Go to the GitHub page of your fork.
#. In the banner with the message "This branch is n commits ahead of openmrslab:master," click on the **Pull request** button.
#. Click on the green **Create pull request** button.
#. The maintainers of OpenMRSLab will review your pull request. If changes are required, repeat Step 3. Once the pull request is approved, you may merge your pull request.