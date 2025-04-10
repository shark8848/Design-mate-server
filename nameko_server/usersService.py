# user_service.py

from nameko.rpc import rpc,RpcProxy
#from nameko.exceptions import PluginException

import hashlib
import sys
sys.path.append("..")
#from apocolib.apocolog4p import apoLogger as apolog
from apocolib.NamekoLogger import namekoLogger as nameko_logger
from apocolib import sqliteSession as sqlSession
from apocolib import ExceptionHandler
from apocolib import ExceptionHandlingProvider
from urpfModel import User

class userService:

    name = 'usersService'
    securityService = RpcProxy('securityService')
#    exception_handler = ExceptionHandler.ExceptionHandler()
#    exception_handling = ExceptionHandlingProvider.ExceptionHandlingProvider()



    @rpc
    def add_user(self, userInfo):#organizationCode, userId, userName, password, role, status):
#        engine = create_engine(get_db_url())
#        Base.metadata.create_all(engine)
#        Session = sessionmaker(bind=engine)
        try:
            organizationCode = userInfo['organizationCode']
            userId = userInfo['userId']
            userName = userInfo['userName']
            password = userInfo['password']
            role = userInfo['role']
            email = userInfo['email']
            phone = userInfo['phone']
            #status = userInfo['status']
            status = '1'

            assert isinstance(organizationCode, str), f'organizationCode must be str, but got {type(organizationCode)}'
            assert isinstance(userId, str), f'userId must be str, but got {type(userId)}'
            assert isinstance(userName, str), f'userName must be str, but got {type(userName)}'
            assert isinstance(password, str), f'password must be str, but got {type(password)}'
            assert isinstance(email, str), f'email must be str, but got {type(email)}'
            assert isinstance(phone, str), f'phone must be str, but got {type(phone)}'
            assert role in ['admin', 'user'], f'role must be either "admin" or "user", but got {role}'
            assert status in ['1', '0'], f'status must be either "1" or "0", but got {status}'

            with sqlSession.sqliteSession().getSession() as session:
            #with getSession() as session:

                user = User(
                    organizationCode=organizationCode,
                    userId=userId,
                    userName=userName,
                    password=encryptPassword(password),
                    role=role,
                    email=email,
                    phone=phone,
                    status=status
                )
                session.add(user)
                session.commit()
                return 0,f'add user {userId} successfully'

        except Exception as e:
            nameko_logger.error(f'add user {userId} error.{str(e)}')
            #raise Exception(f'add user {userId} error.{str(e)}')
            return -1, f'add user {userId} failed: {str(e)}'


    @rpc
    def authenticate(self, userId, password):
        nameko_logger.info(f' request authenticate,{userId},{password}')
        try:
            with sqlSession.sqliteSession().getSession() as session:
            #with getSession() as session:
                user = session.query(User).filter_by(userId=userId, password=encryptPassword(password)).first()
                if user:
                    if user.status =='1':
                        nameko_logger.info('Authentication succeeded')
                        return 0,'Authentication succeeded'
                    else:
                        nameko_logger.info('Authentication failed,Account is disabled')
                        return -2,'Authentication failed,Account is disabled'
                else:
                    nameko_logger.info('Authentication failed,Incorrect username or password')
                    return -1,'Authentication failed,Incorrect username or password'
        except Exception as e:
            nameko_logger.error(f'authenticate user {userId} error.{str(e)}')
            return -1, f'authenticate user {userId} failed: {str(e)}'

    @rpc
    def email_is_registered(self,email):
        nameko_logger.info(f' request email,{email}')
        try:
            with sqlSession.sqliteSession().getSession() as session:
            #with getSession() as session:
                user = session.query(User).filter_by(email=email).first()
                if user:
                    if user.status =='1':
                        nameko_logger.info(f'email {email} is registered')
                        return 0,f'email {email} is registered'
                    else:
                        nameko_logger.info(f'the email {email} ,Account is disabled')
                        return -2,f'the email {email} ,Account is disabled'
                else:
                    nameko_logger.info(f'the email {email} is not registered')
                    return -1,f'the email {email} is not registered'
        except Exception as e:
            nameko_logger.error(f'authenticate email {email} error.{str(e)}')
            return -1, f'authenticate email {email} failed: {str(e)}'

    @rpc
    def auth_mail_token(self,token):
        try:
            # 校验token
            res,data,msg = self.securityService.authenticate(token)
            if res != 0:
                return res,{},msg

            # 校验邮件地址
            res,msg = self.email_is_registered(data['email'])
            if res != 0:
                return res,{},msg
            return 0,data,'mail token is valid'

        except Exception as e:
            nameko_logger.error(f'authenticate email token failed: {str(e)}')
            return -2, f'authenticate email token failed: {str(e)}'

    @rpc
    def edit_user(self, userInfo):#userId, organizationCode=None, userName=None, password=None, role=None, email=None, phone=None, status=None):

        try:

            userId = userInfo['userId']
            organizationCode = userInfo['organizationCode']
            userName = userInfo['userName']
            #password = userInfo['password']
#            password=encryptPassword(userInfo['password'])
            role = userInfo['role']
            email = userInfo['email']
            phone = userInfo['phone']
            status = userInfo['status']
            avator = userInfo['avator']

            assert isinstance(userId, str), f'userId must be str, but got {type(userId)}'
            assert isinstance(organizationCode, str) or organizationCode is None, f'organizationCode must be str or None, but got {type(organizationCode)}'
            assert isinstance(userName, str) or userName is None, f'userName must be str or None, but got {type(userName)}'
#            assert isinstance(password, str) or password is None, f'password must be str or None, but got {type(password)}'
            assert isinstance(email, str) or email is None, f'email must be str or None, but got {type(email)}'
            assert isinstance(phone, str) or phone is None, f'phone must be str or None, but got {type(phone)}'
            assert role in ['admin', 'user'] or role is None, f'role must be either "admin" or "user", but got {role}'
            assert isinstance(avator, str) or avator is None, f'avator must be str or None, but got {avator}'
            assert status in ['1', '0'] or status is None, f'status must be either "1" or "0", but got {status}'

            #with getSession() as session:
            with sqlSession.sqliteSession().getSession() as session:
                user = session.query(User).filter_by(userId=userId).first()
                if user is None:
                    return -1, f'user {userId} is not found'

                if organizationCode is not None:
                    user.organizationCode = organizationCode
                if userName is not None:
                    user.userName = userName
                #if password is not None:
                    #user.password = password
                if role is not None:
                    user.role = role
                if email is not None:
                    user.email = email
                if phone is not None:
                    user.phone = phone
                if avator is not None:
                    user.avator = avator
                if status is not None:
                    user.status = status

                session.commit()
                return 0, f'update user {userId} successfully'

        except Exception as e:
            nameko_logger.error(f'update user {userId} error.{str(e)}')
            return -1, f'update user {userId} failed: {str(e)}'
    #------------------------------------------------------
    @rpc
    def reset_password_by_email(self, email,password):#userId, organizationCode=None, userName=None, password=None, role=None, email=None, phone=None, status=None):

        try:

            password=encryptPassword(password)

            assert isinstance(password, str) or password is None, f'password must be str or None, but got {type(password)}'
            assert isinstance(email, str) or email is None, f'email must be str or None, but got {type(email)}'
            #with getSession() as session:
            with sqlSession.sqliteSession().getSession() as session:
                user = session.query(User).filter_by(email=email).first()
                if user is None:
                    return -1, f'email {email} is not found'

                if password is not None:
                    user.password = password
                if email is not None:
                    user.email = email

                session.commit()
                return 0, f'update user {user.userId} password successfully'

        except Exception as e:
            nameko_logger.error(f'update user {user.userId} error.{str(e)}')

    #------------------------------------------------------
    @rpc
    def delete_user(self, userId):
        try:
            with sqlSession.sqliteSession().getSession() as session:
            #with getSession() as session:

                user = session.query(User).filter_by(userId=userId,status='1').first()
                if user is None:
                    #session.close()
                    return -1,f'user {userId} is not found'

                user.status = '0'
                #session.delete(user)
                session.commit()
                return 0,f'delete user {userId} successfully'

        except Exception as e:
            nameko_logger.error(f'delete user {userId} error.{str(e)}')
            return -1, f'delete user {userId} failed: {str(e)}'

    @rpc
    def get_user_by_id(self, userId):

        try:
            with sqlSession.sqliteSession().getSession() as session:
            #with getSession() as session:

                user = session.query(User).filter_by(userId=userId).first()
                #session.close()
                if user is None:
                    return -1,None,f'get_user_by_id {userId},no data found'
                else:
                    return 0,{
                        'organizationCode': user.organizationCode,
                        'userId': user.userId,
                        'userName': user.userName,
                        #'password': user.password,
                        'role': user.role,
                        'email':user.email,
                        'phone':user.phone,
                        'status': user.status,
                        'avator': user.avator
                    },'get_user_by_id successfully'
        except Exception as e:
            nameko_logger.error(f'get user {userId} error.{str(e)}')
            return -1, {}, f'get user {userId} failed: {str(e)}'

    @rpc
    def get_users_by_organization_code(self, organizationCode):
        try:
            with sqlSession.sqliteSession().getSession() as session:
            #with getSession() as session:

                users = session.query(User).filter_by(organizationCode=organizationCode,status='1').all()
                #nameko_logger.info(f"users info {users} ")
                #session.close()
                if users is None:
                    return -1,[],'get_users_by_organization_code {organizationCode} ,no data found'
                else:
                    return 0,[{
                        'organizationCode': user.organizationCode,
                        'userId': user.userId,
                        'userName': user.userName,
                        #'password': user.password,
                        'role': user.role,
                        'email':user.email,
                        'phone':user.phone,
                        'status': user.status
                    } for user in users],'get_users_by_organization_code {organizationCode} successfully'
        except Exception as e:
            nameko_logger.error(f'get_users_by_organization_code {organizationCode} error.{str(e)}')
            return -1, [],f'get_users_by_organization_code {organizationCode} error.{str(e)}'

def encryptPassword(password):
    encoded_password = password.encode('utf-8')
    hash_object = hashlib.sha256(encoded_password)
    hex_dig = hash_object.hexdigest()
    return hex_dig
