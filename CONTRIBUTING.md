# Contributing to 4C

Thank you for your willingness to contribute to 4C.
The steps outlined in [Setup and Initial Configuration](#setup-and-initial-configuration) have to be performed only
once, while the  [4C Development Workflow](#the-4c-development-workflow) has to be cycled for every bit of code
development in 4C.

**All contributions to 4C must adhere to [The European Code of Conduct for Research Integrity – Revised Edition 2023](http://www.doi.org/10.26356/ECOC).**

4C development is strongly based on the [GitHub Flow](https://guides.github.com/introduction/flow/index.html) which is a
branch-based workflow involving only two types of branches: the `main` branch and `feature` branches.
The most important rules are:

- Anything in the `main` branch is always deployable, i.e. considered stable.
- Development (and bugfixes) are carried out in `feature` branches.

To incorporate a `feature` branch into the `main` branch, 4C employs GitLab's *merge request* mechanism resulting in a
*merge commit*.

### Contents

1. [Setup and Initial Configuration](#setup-and-initial-configuration)
1. [The 4C Development Workflow](#the-4c-development-workflow)
    1. [Create a GitLab Issue](#create-a-gitlab-issue)
    1. [Work an Issue](#work-an-issue)
        1. [Create a Feature Branch](#create-a-feature-branch)
        1. [Make your Changes](#make-your-changes)
        1. [Integrate changes from `main` into your feature branch](#integrate-changes-from-main-into-your-feature-branch)
        1. [Test your Changes](#test-your-changes)
    1. [Merging Changes into `main`](#merging-changes-into-main)
        1. [Push your branch to GitLab](#push-your-branch-to-gitlab)
        1. [Create a Merge Request](#create-a-merge-request)
        2. [Work-in-Progress Merge Requests](#work-in-progress-merge-requests)
        1. [Feedback, Review, and Approval](#feedback-review-and-approval)
        1. [Merging a Merge Request](#merging-a-merge-request)
    1. [Final Clean-Up](#final-clean-up)
    1. [Actively watch the nightly pipeline](#actively-watch-the-nightly-pipeline)

## Setup and Initial Configuration

### Read the [README.md](https://gitlab.lrz.de/baci/baci/blob/main/README.md)

Do ensure you are familiar with all the information in
our [README.md](https://gitlab.lrz.de/baci/baci/blob/main/README.md) file, as that is necessary for understanding what
follows.
Double-check that your local Git repository is configured as described in
the [user documentation](https://baci.pages.gitlab.lrz.de/baci/readthedocs/4Csetup.html#set-up-your-gitlab-account).

### Setup your Integrated Development Environment

We recommend to use an Integrated Development Environment (IDE) for code development because it provides many convenient
features and also eases to comply with our mandatory code style.
Set-up instructions for various IDEs can be found in
the [user documentation](https://baci.pages.gitlab.lrz.de/baci/readthedocs/4Csetup.html#set-up-your-ide).

## The 4C Development Workflow

### Create a GitLab Issue

It is highly recommended to open an issue, when planning any change to 4C. It is good practice to open the issues when
starting to work on something (and not after working on it or just prior to create the merge request).

Navigate to 4C's [GitLab Issues page](https://gitlab.lrz.de/baci/baci/issues) and create a new issue.
The issue can be used for any number of things &mdash; reporting a bug, suggesting an enhancement, posing a question,
etc.
On the new issue creation page, select an issue template from the drop down menu
to pre-populate the *Description* field with some text.
Follow the instructions in that template to give your fellow developers as much information as you can
such that the issue can be understood and tackled as soon as it is practicable.

Make sure to assign appropriate team::<teamname> label(s) according to the issue type and @mention the respective teams
in order to facilitate your fellow developers to keep track of the issues that are related to their field.

Issues begin their life in the **Open** column of our [issue board](https://gitlab.lrz.de/baci/baci/-/boards)
and then move through the board from left to right. Each column represents the current state of the issue as indicated
by the status label.
If at any point in an issue's life it becomes blocked by something (either another 4C issue, or perhaps something
external),
move the issue card into the **Blocked** column to indicate that work can't proceed until something else is dealt with.
Add a comment to the blocked issue to explain why it is blocked and what needs to happen to resolve the **Blocked**
status.

[↑ Contents](#contents)

### Work an Issue

When work commences on an issue, move the issue card to the **In Progress** column of
our [issue board](https://gitlab.lrz.de/baci/baci/-/boards).
Then the workflow to use is the following:

#### Create a Feature Branch

> **Note:** It is important to keep your local `main` branch in 4C up-to-date with the remote `main`. Hence,
> creation of a feature branch consists of *two* steps.

First, update the `main` branch:

```bash
cd <path/to/four-c-source-code>
git checkout main
git pull
```

where `<path/to/four-c-source-code>` is the location of your local 4C repository, i.e. the 4C source code.

Then, create a local branch off of `main` in 4C on which to make your changes:

```bash
git branch <branchName>
git checkout <branchName>
```

The branch name `<branchName>` can be whatever you like, though we have some recommendations:

* Include the issue number in it in some way, for instance, `123-<restOfBranchName>`, or `<restOfBranchName>-123`.
* Make the branch name descriptive; that is, avoid `fixSomeStuff`, `performanceTweaks`, and generic names along those
  lines.
* To indicate your branch is intended solely for your own use, include your username in the branch name somehow, as
  in `<username>-<restOfBranchName>` or `<restOfBranchName>-<username>`.

> **Note:** Use `git branch` to list all available branches.

[↑ Contents](#contents)

#### Make your Changes

* Do whatever work is necessary to address the issue you're tackling.
* Commit your changes locally.

> **Note:** Break your work into logical, compilable commits.
> Feel free to commit small chunks early and often in your local repository and
> then use `git rebase -i` to reorganize your commits before sharing.

[↑ Contents](#contents)

##### Commit Messages

Use the commit message to explain the context and reasons for your changes, i.e. the **What?** and **Why?**, rather than
providing details on how you changed the code. Use comments in the source code or in the Doxygen documentation to
explain *how* the code works.

Commit messages must be detailed and directly include all necessary information. Including references to issues and
merge requests is considered good practice, but does not replace a detailed and self-contained commit message.

To differentiate from the documentation in GitLab: Commit messages are for the developers of the affected code parts,
whereas issue and merge requests descriptions are meant for reviewers.

Commit messages should meet the following points:

* The **first line** of the commit message should be a descriptive title, **limited to 50 characters**.
* This is then followed by a blank line, and then the rest of the commit message is a description of the changes,
  limited to 72 characters wide.
* Make sure your commit messages reference the appropriate GitLab issue numbers using the `#<issueNumber>` syntax.

[↑ Contents](#contents)

##### Doxygen

4C uses [Doxygen](https://www.doxygen.nl/) to generate documentation from annotated source code.
Please see
the [user documentation](https://baci.pages.gitlab.lrz.de/baci/readthedocs/doxygen.html#documenting-the-code-with-doxygen)
for our Doxygen guidelines.

[↑ Contents](#contents)

#### Integrate changes from `main` into your feature branch

While working on your feature in your local `<branchName>` branch in 4C, other commits will likely make it into the
remote `main` branch. There are a variety of ways to incorporate these changes into your local feature branch. Our
preferred possibility is a rebase onto the newest available `main`, e.g. by running:

```bash
git fetch
git rebase -i origin/main
```

For more information on `git rebase` you may want to check
out [the documentation](https://git-scm.com/book/en/v2/Git-Branching-Rebasing).

> **Note:** It might happen that conflicts arise during the `git rebase` operation. After seeing a conflict, you can do
> two things:
>
> * Decide not to rebase. Run `git rebase --abort` to abort the rebase operation and to restore your version of the
    code (without incorporating changes from `main`).
> * Resolve the conflicts. Git will mark the conflicts in the working tree. Edit the files into shape and `git add` them
    to the index. The editing is especially simple if you run `git mergetool`, which guides you through conflict
    resolution. Use `git commit` to seal the deal.

> **Note:** You might want to integrate changes from `main` in this fashion on a regular basis to ease resolving of
> possible conflicts.

[↑ Contents](#contents)

#### Test your Changes

To ensure your changes haven't broken anything, run `ctest` in your 4C build directory.
A small set of test cases can be run via `ctest -L minimal`.

[↑ Contents](#contents)

### Merging Changes into `main`

To bring changes form a locally developed feature branch into `main` and make them available to everyone, 4C relies on
GitLab's _merge request (MR)_ mechanism. 4C's MR workflow is outlined in the following.

#### Push your branch to GitLab

To publish your changes and make them available to others, you have to push them to GitLab. Before pushing your branch
to GitLab, use interactive rebasing via `git rebase -i` to squash the commits on your feature branch into the smallest
number of logical commits. Much of the time this will just mean a single commit, but you may wish to keep more than one
&mdash; for instance, have the majority of your feature addition in one commit, but keep some performance tweaks
separate in another commit, in case it becomes necessary to revert the performance tweaks later while keeping the rest
of the feature.

Push your local feature branch up to the remote with:

```bash
git push --set-upstream origin <branchName>
```

#### Create a Merge Request

When your changes are ready to be integrated into 4C's `main` branch,
move the issue card from **In Progress** to **Under Review** on our
[issue board](https://gitlab.lrz.de/baci/baci/-/boards) and then:

* Navigate to the 4C project on GitLab
  and [create a new merge request](https://gitlab.lrz.de/baci/baci/merge_requests/new):
    * Be sure you choose:
        * source branch: `<branchName>`
        * target branch: `main`
    * On the new merge request creation page, select a merge request template from the dropdown menu to pre-populate the
      *Description* field with some text. Follow the instructions in that template to give as much information as you
      can such that the merge request can be reviewed and merged as soon as it is practicable.

  > **Note** Do not create the merge request by clicking on the button *Create merge request* in your issue (creates a
  new branch).

    * To notify others about your merge request, @mention affected teams and other interested parties in the *Interested
      Parties / Possible Reviewers* section of the *Description* field.
    * Assign some of your fellow developers as *Reviewers* to request their review and **required** approval.
    * Assign the MR to yourself, as you are probably the person that will work the most on this MR.

After opening the MR, a testing pipeline is automatically created. All test jobs need to pass before the MR can be
merged.

[↑ Contents](#contents)

##### Work-in-Progress Merge Requests

If work on an issue is not yet complete, but you'd like to get another set of eyes on your work sooner rather than
later,
you can create a "work-in-progress" merge request.
Simply begin the *Title* with `WIP:` and that will indicate to everybody that this is ongoing work
that is not necessarily meant for review yet.

If you are working on a feature addition that is fairly substantial (say greater than a month of work),
consider creating a WIP merge request. These can be reviewed, but then you can close them without merging in the
changes.
When work is complete, create a merge request that includes all the changes,
and mention all the sub-WIP-merge-requests that have already been reviewed in the *Description*.
This makes it easy for a reviewer to see that all the changes have already been reviewed along the way,
rather than having to look at the entire change set at once.

[↑ Contents](#contents)

#### Feedback, Review, and Approval

At this point you'll enter into a stage where you and various 4C developers will iterate back and forth until your
changes are in an acceptable state and can be merged in.
If you need to make changes to your merge request, make additional commits on your `<branchName>` branch and push them
up to the remote.
If the changes are minor changes (typos, renames, etc.) compared to the original commits, you can add them to the
respective commits, to keep the commit log meaningful.
Make sure you don't delete your remote feature branch before your merge request has been merged.

> Independent of your level of expertise, experience with 4C, employment status, or whatsoever, you are encouraged to
> actively participate in this process _in the best interest of collaborative code development_. **Every contribution is
valuable!**

To stimulate an open and constructive discussion and to get the MR ready to be merged, this includes:

- _Anybody_ can make comments and/or raise questions about the proposed changes.
- Code owners have to approve the changes to "their" files, when they agree with them.
- The MR author will actively join the discussion and respond to comments and questions.

> As always: Be constructive! Appreciate each others work and feedback!

Some remarks on code review:

- Code review is intended to be a constructive discussion about the **Why?** and **How?** of the proposed changes.
- Recognizing the fact, that there might be more than one good solution to a given problem, code review is intended to
  increase overall code quality and hopefully detect some critical points before merging.
- We have a few mandatory [coding guidelines](https://baci.pages.gitlab.lrz.de/baci/readthedocs/codingguidelines.html#)
  in place, that need to be enforced during code review.

[↑ Contents](#contents)

#### Merging a Merge Request

To merge changes into `main`, a feature branch needs to satisfy the following conditions:

* Passing code check, e.g. no trailing white spaces, proper Doxygen style, ...
* No build errors and warnings
* All tests are passing.
* Two approvals are given.

Before the merge is performed, consider a rebase of your branch into a final clean state.
For instance, if you added small
modifications during review these should be squashed into the appropriate commits, as the exact history of these changes
is not of interest in the long run and pollutes the log.

When these conditions are met,
the merge can be triggered using the "Merge" button on the merge request page on GitLab.
It is good practice that the MR author triggers the merge,
since they are in the best position to decide, whether they want to merge right now or want to include more changes in
this MR.

> Yet, basically every @baci/baci_developers can press the "Merge" button.

If your merge request *Description* has some form of "closes #\<issueNumber\>" in it somewhere, merging the merge
request will automatically close the associated issue, which will move the issue card from **Under Review** to **Done**
on the [issue board](https://gitlab.lrz.de/baci/baci/-/boards). If not, you'll need to make this move manually and adapt
each issues' labels manually.

[↑ Contents](#contents)

### Final Clean-Up

To keep the repository clean, delete your feature branch *after* merging it into `main`. When you merge a merge
request, GitLab will give you the option to click a button to remove the source branch. If you click this, then

```bash
git fetch --prune
```

will remove the remote tracking information from your local repository. Alternatively, you could skip the GitLab button
and use

```bash
git push origin --delete <branchName>
```

Either way is completely fine. After that you can remove your local branch with

```bash
git branch -D <branchName>
```

[↑ Contents](#contents)

### Actively watch the nightly pipeline

After your merge request was merged into the `main` branch, actively watch the nightly pipeline which is executed in
the following night.
In the case any configuration on the `main` branch of the nightly pipeline is failing (this can happen as we do not
test the same/all configurations before the merge):

* All **developer(s)** and **reviewer(s)** who merged into the `main` branch since the last nightly pipeline passed
  are **responsible** for fixing the reason for the failing pipeline.
* This responsible group coordinates this bug-fixing process independently.
* If you are member of the responsible group, open an issue using the `TEST_FAILING` template if not yet done (can be
  skipped for immediately fixed trivial bugs).
* The code owners and other developers support this process if requested but are not leading this process.
* If a merge request has to be reverted is decided by the reviewer (in discussion with the developer), depending on the
  expected time to fix the issue and how critical the problem is.

[↑ Contents](#contents)
