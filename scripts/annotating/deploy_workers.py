import asyncio

from .modal_grounding import GroundingModalWrapper as WrapperCls


async def deploy_workers():
    wrapper = WrapperCls()
    # The workers are automatically deployed when you interact with them
    print("Deploying Wrapper workers...")


if __name__ == "__main__":
    asyncio.run(deploy_workers())
