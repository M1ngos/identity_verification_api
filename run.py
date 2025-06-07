# Copyright (c) 2025 ITS
#
# All rights reserved.
#
# This software is proprietary and confidential. Unauthorized copying, use,
# modification, or distribution of this software is strictly prohibited unless
# explicitly authorized by ITS.

import uvicorn

if __name__ == "__main__":
    # Run the server
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)