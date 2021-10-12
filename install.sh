#!/usr/bin/env sh

global_flag='--global'
if [ $# -gt 1 ] || [ $# -eq 1 ] && [ "$1" != "$global_flag" ]; then
    printf 'Lambeq installer.\n\nUsage:\n'
    printf '  install.sh %-9s Run the interactive installer.\n' ''
    printf '  install.sh %-9s Install globally without prompts (not recommended).\n' "$global_flag"
    exit
elif [ $# -eq 0 ]; then
    cat <<EOF
Lambeq installer
----------------
This script is primarily for installing Lambeq with depccg, but can also be
used in the following ways:

    1. Install Lambeq and depccg only.
    2. Install base Lambeq only, no extra features.
    3. Install Lambeq with all extra features (including depccg).

EOF
    printf 'Choose an option (default: 1): [1-3] '
    read -r install_option

    if [ "${install_option:=1}" != 1 ] && [ "$install_option" != 2 ] && [ "$install_option" != 3 ]; then
        echo 'Invalid response. Exiting.'
        exit
    fi

    if [ "$VIRTUAL_ENV" ]; then
        echo 'Virtual environment detected.'
        option_text="Install in current virtual environment '$VIRTUAL_ENV'"
    else
        option_text='Install globally (not recommended)'
    fi

    cat <<EOF

Choose a location to install Lambeq:
    1. Install in a new virtual environment.
    2. $option_text.

EOF
    printf 'Choose an option (default: 1): [1/2] '
    read -r answer

    if [ "${answer:-1}" = 1 ]; then
        printf 'Enter location for virtual environment: '
        read -r venv
        if [ ! "$venv" ]; then
            echo 'No location entered. Exiting.'
            exit
        elif [ -d "$venv" ]; then
            printf '\n'%s' exists.\n\n' "$venv"
            printf 'Overwrite with new virtual environment (not recommended)? [y/N] '
            read -r answer
            if [ "$answer" = "${answer#[Yy]}" ]; then exit; fi
        fi

        echo "Creating virtual environment at '$venv'..."
        if ! python3 -m venv "$venv"; then
            echo "Failed to create virtual environment at '$venv'. Exiting."
            exit
        fi

        . "$venv/bin/activate"
    elif [ "$answer" != 2 ]; then
        echo 'Invalid response. Exiting.'
        exit
    fi
else
    install_option=3  # default for non-interactive (i.e. global) install
fi

# determine installation source
lambeq="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
if [ ! -r "$lambeq/pyproject.toml" ]; then
    lambeq='lambeq'
fi

echo 'Preparing installation environment...'
python3 -m pip install --upgrade pip wheel
if [ "$install_option" != 2 ]; then
    python3 -m pip install cython numpy

    if [ "$install_option" = 1 ]; then
        extras='depccg'
    else
        extras='all,test'
    fi

    python3 -m pip install --use-feature=in-tree-build "$lambeq""[$extras]"

    echo 'Downloading pre-trained depccg parser...'
    python3 -m depccg en download
else
    echo 'Installing base Lambeq...'
    python3 -m pip install --use-feature=in-tree-build "$lambeq"
fi

echo 'Installation complete.'
if [ "$venv" ]; then
    echo "To use Lambeq, activate the virtual environment at '$venv'."
fi
