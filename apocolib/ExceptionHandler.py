import sys
sys.path.append("..")
from apocolib.NamekoLogger import namekoLogger as nameko_log
from nameko.extensions import DependencyProvider

class ExceptionHandler(DependencyProvider):
    def worker_setup(self, worker_ctx):
        worker_ctx.data['logger'] = nameko_log

    def worker_result(self, worker_ctx, result=None, exc_info=None):
        print(f'test {exc_info}')

        if exc_info is not None:
            nameko_log.error(
                "Error in {}:{} - {}".format(
                    worker_ctx.service_name,
                    worker_ctx.entrypoint.method_name,
                    str(exc_info[1])
                )
            )
