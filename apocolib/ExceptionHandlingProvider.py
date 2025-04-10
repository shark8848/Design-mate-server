from nameko.rpc import RpcProxy
import sys
sys.path.append("..")
from apocolib.NamekoLogger import namekoLogger as nameko_log
from nameko.extensions import DependencyProvider

class ExceptionHandlingProvider(DependencyProvider):
    def __init__(self):
        pass
        #self.rpc_proxy = RpcProxy()

    def handle_exception(self, worker_ctx, exc_info):
        # call another service using RpcProxy
        #self.rpc_proxy.my_other_service.my_method(str(exc_info))
        # handle the exception here
        print(f"An exception occurred: {str(exc_info)}")

    def worker_setup(self, worker_ctx):
        worker_ctx.data['logger'] = nameko_log
        worker_ctx.handle_result = self.handle_result
        worker_ctx.handle_exception = self.handle_exception

    def handle_result(self, worker_ctx, result, exc_info):
        print(f'test {exc_info}')
        # continue executing the original function code
        if exc_info is not None:
            nameko_log.error(
                "Error in {}:{} - {}".format(
                    worker_ctx.service_name,
                    worker_ctx.entrypoint.method_name,
                    str(exc_info[1])
                )
            )
        print("The function completed successfully.")
        return result
