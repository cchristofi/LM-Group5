<<<<<<< Updated upstream
#!/bin/bash

set -e

PROJECT_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/"
=======
#!/bin/bash

set -e

PROJECT_FOLDER="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )/"
>>>>>>> Stashed changes
exec docker run --rm -it -v "${PROJECT_FOLDER}:/root/projects/" --net=host cigroup/learning-machines:python3 bash